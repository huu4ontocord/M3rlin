import ast
import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

try:
    import wandb
except ImportError:
    wandb = None

from .distributed import is_master
from .precision import get_autocast


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def sample_chunk(chunk, seq_len):
    # if the start_idx changes, we need to return it so we can figure out where the embeddings will be
    if chunk.shape[1] == seq_len + 1:
        start_idx = 0
    elif chunk.shape[1] > seq_len + 1:
        start_idx = torch.randint(0, chunk.shape[1] - seq_len + 1, (1,)).item()
    else:
        raise Exception(f"Invalid sequence length: Sequence length {seq_len} > {chunk.shape[1]} Chunk size")
    
    # HHN Note to self:
    # shift by one autoregressive prediction
    inputs = chunk[:, start_idx:start_idx+seq_len-1]
    targets = chunk[:, start_idx+1:start_idx+seq_len]
    #HHN change
    return inputs, targets, start_idx


def train_one_epoch(
    model, data, loss, epoch, optimizer, scaler, scheduler, args, tb_writer=None
):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)

    model.train()

    data["train"].set_epoch(
        epoch
    )  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data["train"].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    losses_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()

    # used only if --log-logit-mean flag is passed
    logit_m = AverageMeter()

    end = time.time()

    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        if not args.skip_scheduler:
            scheduler(step)

        # CHANGE FOR INTERLEAVED EMBEDDING
        (texts,) = batch
        texts, embeddings, embedding_positions = extract_embeddings_from_texts(texts)
        texts = torch.LongTensor(texts).to(device)
        embeddings = embeddings.to(device)
      
        data_time_m.update(time.time() - end)
        optimizer.zero_grad()
        
        if args.accum_freq == 1:
            with autocast():
                inputs, targets, start_idx = sample_chunk(texts, args.seq_len)
                #HHN
                out, _, out_embeddings = model(inputs, embeddings, embedding_positions) 
                if args.log_logit_mean:
                    logit_m.update(torch.mean(out).item())                  

                #HHN
                # we need to mask out prediction of the <IMG> tokens here for this loss.
                # this would be mask_positions = [(a, b, c+1) for a,b,c in embedding_positions]
                total_loss = loss(out.reshape(-1, args.vocab_size), targets.reshape(-1))
                # total_loss  += alpha * mse_loss(embeddings, out_embeddings) / args.accum_freq
                
            backward(total_loss, scaler)
        else:
            # split up batch into accum_freq chunks -- if you have --batch-size 8 and --accum-freq 4
            # then you only process 2 items at a time. batch-size must be divisible by accume-freq.
            assert (
                args.batch_size % args.accum_freq == 0
            ), "Batch size must be divisible by accum_freq"
            per_batch = args.batch_size // args.accum_freq

            inputs, targets, start_idx = sample_chunk(texts, args.seq_len)

            for ii in range(args.accum_freq):
                with autocast():
                    inputs_ii = inputs[ii * per_batch : (ii + 1) * per_batch]
                    targets_ii = targets[ii * per_batch : (ii + 1) * per_batch]

                    #HHN
                    #embedding_positions = [(a, b, c+start_idx) for a,b,c in embedding_positions]
                    #embeddings = embeddings[:,:,start_idx]
                  
                    out, _, out_embeddings = model(inputs_ii, embeddings_ii, embedding_positions_ii) 
                    
                    if args.log_logit_mean:
                        logit_m.update(torch.mean(out).item())
                    # we need to mask out prediction of the <IMG> tokens here for this loss.
                    # this would be mask_positions = [(a, b, c+1) for a,b,c in embedding_positions]
                    local_loss = (
                        loss(out.reshape(-1, args.vocab_size), targets_ii.reshape(-1))
                        / args.accum_freq
                    )
                    # local_loss += alpha * mse_loss(embeddings, out_embeddings) / args.accum_freq
              
                backward(local_loss, scaler)
                if ii == 0:
                    total_loss = local_loss
                else:
                    total_loss += local_loss

        if scaler is not None:
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip_norm, norm_type=2.0
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                if isinstance(model, FSDP):
                    model.clip_grad_norm_(args.grad_clip_norm, norm_type=2.0)
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip_norm, norm_type=2.0
                    )
            optimizer.step()

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if is_master(args) and (
            i % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch
        ):
            batch_size = len(inputs)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # gathered_loss = [torch.zeros_like(total_loss) for _ in range(args.world_size)]
            # torch.distributed.all_gather(gathered_loss, total_loss)
            # losses_m.update(sum(gathered_loss).item() / args.world_size, batch_size * args.world_size)
            losses_m.update(total_loss.item(), batch_size)
            samples_per_second = inputs.numel() * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = inputs.numel() / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {losses_m.avg:.3f} "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": losses_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "lr": optimizer.param_groups[0]["lr"],
                "tokens": (step + 1) * args.batch_size * args.seq_len * args.world_size,
            }

            if args.log_logit_mean:
                log_data["logit_mean"] = logit_m.val

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, "Please install wandb."
                    wandb.log({name: val, "step": step, "tokens": log_data["tokens"]})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()

            if math.isnan(losses_m.val):
                # case where loss goes to nan, we see this sometimes with bad nodes.
                # in this case we would like to free resources and prevent other issues
                # e.g., saving checkpoints and optmization states that may lead to skipped
                # training on restarts.
                return False

    # end for
    return True
  
def extract_embeddings_from_texts(texts):
  # HHN - todo
  # got through the batch and the rows and and each item, 
  # We need to find all the non ints here, and put in a special token, <IMG>
  # create the embeddings from the data where there were non-ints
  # save away the position and triples
  return texts, embeddings, positions

def evaluate(model, data, start_epoch, args, writer):
    """
    evaluates perplexity on validation data
    """
    if is_master(args):
        print("=> begin evaluation")
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)

    model.eval()

    data["val"].set_epoch(
        start_epoch
    )  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data["val"].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    losses_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    sps_m = AverageMeter()
    spspg_m = AverageMeter()
    end = time.time()
    loss = torch.nn.CrossEntropyLoss()
    for i, batch in enumerate(dataloader):
        # CHANGE FOR INTERLEAVED EMBEDDING
        (texts,) = batch
        # We need to find all the non ints here, and put in a special token, <IMG>, etc. . 
        # and keep around a dict of the position and the embedding to feed into model.
        texts, embeddings, embedding_positions = extract_embeddings_from_texts(texts)
        texts = torch.LongTensor(texts).to(device)
        embeddings = embeddings.to(device)
      
        data_time_m.update(time.time() - end)

        with autocast():
            inputs, targets, start_idx = sample_chunk(texts, args.seq_len)
            #shift this is probably wrong, but we need to the column by start_idx
            #embedding_positions = [(a, b, c+start_idx) for a,b,c in embedding_positions]
            #embeddings = embeddings[:,:,start_idx]
            out, _, out_embeddings = model(inputs, embeddings, embedding_positions) 
            # we need to mask out prediction of the <IMG> tokens here for this loss.
            # this would be mask_positions = [(a, b, c+1) for a,b,c in embedding_positions]
            total_loss = loss(out.reshape(-1, args.vocab_size), targets.reshape(-1))
            # now we need to do MSE Loss for prediction 
            # total_loss += alpha * mse_loss(embeddings, out_embeddings), 
            losses_m.update(total_loss.item(), inputs.shape[0])
        batch_time_m.update(time.time() - end)
        sps_m.update(inputs.numel() * args.world_size / batch_time_m.val)
        spspg_m.update(inputs.numel() / batch_time_m.val)

    # Save eval loss / etc.
    log_data = {
        "loss": losses_m.avg,
        "data_time": data_time_m.avg,
        "batch_time": batch_time_m.avg,
        "samples_per_second": sps_m.avg,
        "samples_per_second_per_gpu": spspg_m.avg,
        "tokens": start_epoch * args.train_num_samples * args.seq_len,
    }

    for name, val in log_data.items():
        name = "valid/" + name
        if writer is not None:
            writer.add_scalar(name, val, start_epoch)
        if args.wandb and is_master(args):
            assert wandb is not None, "Please install wandb."
            wandb.log({name: val, "epoch": start_epoch, "tokens": log_data["tokens"]})
    if is_master(args):
        print(f"evaluation perplexity: {math.exp(losses_m.avg)}")
    return log_data
