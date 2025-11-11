#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
High-throughput preprocessor for engage_seq_v4 JSONL files.

Design:
- Multi-threaded streaming JSONL file reading with thread pool
- Multi-process workers for data processing
- Filters users by minimum sequence length
- Condenses to essential fields (ad_id, conversion_timestamp)
- Splits into train/eval datasets
- Direct JSONL output from workers

Example usage:
    python preprocess_engage_seq.py \
        --input_dir ~/local/data/engage_seq_v4 \
        --output_dir tmp/engage_seq_v4 \
        --min_seq_len 10 \
        --num_workers 16 \
        --num_reader_threads 100 \
        --batch_size 10000
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import signal
import sys
from concurrent.futures import as_completed, ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Set

import orjson

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s[%(levelname)s]%(filename)s:%(lineno)d:%(message)s",
)
logger = logging.getLogger(__name__)

SENTINEL = "__SENTINEL__"


def _read_single_file_worker(
    file_path: Path,
    batch_size: int,
    in_q: mp.Queue,
) -> int:
    """Worker function to read a single JSONL file and put batches into queue."""
    buf: List[bytes] = []
    local_count = 0

    with open(file_path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            buf.append(line)
            local_count += 1

            # Yield batch when buffer is full
            if len(buf) >= batch_size:
                in_q.put(buf)
                buf = []

    # Put any remaining documents in buffer
    if buf:
        in_q.put(buf)

    logger.info(f"Completed reading {file_path.name}: {local_count} records")
    return local_count


def iter_json_batches_from_folder(
    input_dir: str,
    batch_size: int,
    in_q: mp.Queue,
    split: str,
    train_split_ratio: float,
    num_reader_processes: int = 8,
) -> None:
    """Read JSONL files from folder using multi-processing and put batches into queue."""
    jsonl_files = sorted(Path(input_dir).glob("user_*.jsonl"))

    if not jsonl_files:
        raise ValueError(f"No user_*.jsonl files found in {input_dir}")

    logger.info(f"Found {len(jsonl_files)} JSONL files in {input_dir}")

    # Determine file split based on train_split_ratio
    total_files = len(jsonl_files)
    if split == "train":
        # Take first train_split_ratio% of files for training
        end_idx = int(total_files * train_split_ratio)
        jsonl_files = jsonl_files[:end_idx]
        logger.info(
            f"Processing {len(jsonl_files)} files for train split ({train_split_ratio*100:.0f}% of files)"
        )
    elif split == "eval":
        # Take remaining files for eval
        start_idx = int(total_files * train_split_ratio)
        jsonl_files = jsonl_files[start_idx:]
        logger.info(
            f"Processing {len(jsonl_files)} files for eval split ({(1-train_split_ratio)*100:.0f}% of files)"
        )

    # Use ProcessPoolExecutor for multi-process file reading
    with ProcessPoolExecutor(max_workers=num_reader_processes) as executor:
        # Submit all files to worker processes
        futures = []
        for file_path in jsonl_files:
            future = executor.submit(
                _read_single_file_worker,
                file_path,
                batch_size,
                in_q,
            )
            futures.append(future)

        # Wait for all workers to complete and handle any exceptions
        total_processed = 0
        for future in as_completed(futures):
            try:
                count = future.result()  # This will raise any exception from the worker
                total_processed += count
                if total_processed % 100000 == 0:
                    logger.info(f"[reader] Processed {total_processed} records")
            except Exception as e:
                logger.error(f"Reader worker error: {e}")

    logger.info(
        f"[reader] Completed processing {total_processed} total records for {split} split"
    )


def worker_loop_pass1(
    in_q: mp.Queue,
    output_file: str,
    worker_id: int,
) -> None:
    """Pass 1: Collect all unique ad_tokens in this worker."""

    batches_processed = 0
    total_docs = 0
    local_ad_tokens: Set[str] = set()

    while True:
        batch = in_q.get()

        if isinstance(batch, str) and batch == SENTINEL:
            logger.info(
                f"[Pass1 worker {worker_id}] Shutting down. Processed {batches_processed} batches, "
                f"{total_docs} docs, collected {len(local_ad_tokens)} unique tokens"
            )
            break

        batches_processed += 1
        batch_docs = len(batch)
        logger.info(
            f"[Pass1 worker {worker_id}] Got {batch_docs} docs, queue length = {in_q.qsize()}"
        )
        total_docs += batch_docs

        # Process each record in the batch
        for line in batch:
            try:
                data = orjson.loads(line)

                # Extract necessary fields
                ad_ids = data.get("ad_id", [])
                timestamps = data.get("conversion_timestamp", [])
                ad_token_maps: Dict[str, Any] = data.get("ads_tokenization", {})

                if not (len(timestamps) == len(ad_ids) and len(ad_ids) > 0):
                    continue

                seen_ads = set()
                for timestamp, ad_id in sorted(zip(timestamps, ad_ids), reverse=True):
                    if str(ad_id) not in ad_token_maps or ad_id in seen_ads:
                        continue
                    seen_ads.add(ad_id)
                    ad_token = "_".join(list(map(str, ad_token_maps.get(str(ad_id)))))
                    local_ad_tokens.add(ad_token)

            except Exception as e:
                logger.warning(
                    f"[Pass1 worker {worker_id}] Error processing record: {e}"
                )
                import traceback

                traceback.print_exc()
                break

        if batches_processed % 5 == 0:
            logger.info(
                f"[Pass1 worker {worker_id}] Batch {batches_processed}: {batch_docs} docs processed, "
                f"{len(local_ad_tokens)} unique tokens so far"
            )

    # Save local ad_tokens to output file
    with open(output_file, "w") as out_f:
        for token in local_ad_tokens:
            out_f.write(token + "\n")

    logger.info(
        f"[Pass1 worker {worker_id}] Saved {len(local_ad_tokens)} tokens to {output_file}"
    )


def worker_loop_pass2(
    in_q: mp.Queue,
    output_file: str,
    min_seq_len: int,
    max_seq_len: int,
    worker_id: int,
    ad_token_to_id: Dict[str, int],
) -> None:
    """Pass 2: Process batches of records and write to output file using global ad_token_to_id mapping."""

    batches_processed = 0
    total_docs = 0
    valid_docs = 0
    unique_ad_ids: Set[int] = set()

    # Open output file for this worker
    with open(output_file, "w") as out_f:
        while True:
            batch = in_q.get()

            if isinstance(batch, str) and batch == SENTINEL:
                logger.info(
                    f"[Pass2 worker {worker_id}] Shutting down. Processed {batches_processed} batches, "
                    f"{total_docs} docs, {valid_docs} valid docs"
                )
                break

            batches_processed += 1
            batch_docs = len(batch)
            logger.info(
                f"[Pass2 worker {worker_id}] Got {batch_docs} docs, queue length = {in_q.qsize()}"
            )
            total_docs += batch_docs

            # Process each record in the batch
            for line in batch:
                try:
                    data = orjson.loads(line)

                    # Extract necessary fields
                    ad_ids = data.get("ad_id", [])
                    timestamps = data.get("conversion_timestamp", [])
                    ad_token_maps: Dict[str, Any] = data.get("ads_tokenization", {})

                    # Sort by timestamp (chronological order), dedup and map to 0-index
                    if not (len(timestamps) == len(ad_ids) and len(ad_ids) > 0):
                        continue

                    valid_pairs = []
                    seen_ads = set()
                    for timestamp, ad_id in sorted(
                        zip(timestamps, ad_ids), reverse=True
                    ):
                        if str(ad_id) not in ad_token_maps or ad_id in seen_ads:
                            continue
                        seen_ads.add(ad_id)
                        ad_token = "_".join(
                            list(map(str, ad_token_maps.get(str(ad_id))))
                        )
                        if ad_token in ad_token_to_id:
                            valid_pairs.append((timestamp, ad_token_to_id[ad_token]))

                    if not valid_pairs:
                        continue

                    valid_pairs = sorted(valid_pairs)
                    timestamps, mapped_ad_ids = zip(*valid_pairs)
                    mapped_ad_ids = list(mapped_ad_ids)
                    timestamps = list(timestamps)

                    # Filter: minimum sequence length
                    if len(mapped_ad_ids) < min_seq_len:
                        continue

                    # Apply max sequence length if specified
                    if max_seq_len and len(mapped_ad_ids) > max_seq_len:
                        mapped_ad_ids = mapped_ad_ids[-max_seq_len:]
                        timestamps = timestamps[-max_seq_len:]

                    # Create condensed record
                    condensed_record = {
                        "ad_id": mapped_ad_ids,
                        "conversion_timestamp": timestamps,
                    }

                    # Write to output file
                    out_f.write(json.dumps(condensed_record) + "\n")
                    valid_docs += 1
                    unique_ad_ids.update(mapped_ad_ids)

                except Exception as e:
                    logger.warning(
                        f"[Pass2 worker {worker_id}] Error processing record: {e}"
                    )
                    import traceback

                    traceback.print_exc()
                    break

            if batches_processed % 5 == 0:
                logger.info(
                    f"[Pass2 worker {worker_id}] Batch {batches_processed}: {batch_docs} docs processed, "
                    f"{valid_docs} valid so far"
                )

            logger.info(
                f"[Pass2 worker {worker_id}] Finished {batch_docs} docs, queue length = {in_q.qsize()}"
            )

    # Write worker statistics
    stats = {
        "worker_id": worker_id,
        "total_docs": total_docs,
        "valid_docs": valid_docs,
        "unique_ad_ids": len(unique_ad_ids),
        "max_ad_id": max(unique_ad_ids) if unique_ad_ids else 0,
    }
    logger.info(f"[Pass2 worker {worker_id}] Final stats: {stats}")
    return stats


def collect_statistics(output_dir: str, split: str, num_workers: int) -> Dict[str, Any]:
    """Collect statistics from all worker output files."""
    split_dir = os.path.join(output_dir, split)

    # Merge all worker files into a single output file
    output_file = os.path.join(output_dir, f"{split}.jsonl")
    total_users = 0
    all_ad_ids: Set[int] = set()

    with open(output_file, "w") as out_f:
        for worker_id in range(num_workers):
            worker_file = os.path.join(split_dir, f"worker_{worker_id}.jsonl")
            if os.path.exists(worker_file):
                with open(worker_file, "r") as in_f:
                    for line in in_f:
                        out_f.write(line)
                        total_users += 1
                        # Parse to collect ad_ids for statistics
                        try:
                            data = json.loads(line)
                            all_ad_ids.update(data.get("ad_id", []))
                        except:
                            pass
                # Remove worker file after merging
                os.remove(worker_file)

    logger.info(f"[{split}] Merged {num_workers} worker files into {output_file}")
    logger.info(f"[{split}] Total users: {total_users}")
    logger.info(f"[{split}] Unique ad IDs: {len(all_ad_ids)}")
    logger.info(f"[{split}] Max ad ID: {max(all_ad_ids) if all_ad_ids else 0}")

    return {
        "total_users": total_users,
        "unique_ad_ids": len(all_ad_ids),
        "max_ad_id": max(all_ad_ids) if all_ad_ids else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess engage_seq_v4 JSONL files")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="~/local/data/engage_seq_v4",
        help="Input directory containing user_*.jsonl files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tmp/engage_seq_v4",
        help="Output directory for processed files",
    )
    parser.add_argument(
        "--min_seq_len",
        type=int,
        default=10,
        help="Minimum sequence length to keep a user",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=None,
        help="Maximum sequence length (None = no limit)",
    )
    parser.add_argument(
        "--train_split", type=float, default=0.99, help="Train split ratio (0.0-1.0)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(1, mp.cpu_count() - 1),
        help="Number of worker processes",
    )
    parser.add_argument(
        "--num_reader_processes",
        type=int,
        default=16,
        help="Number of reader processes for file I/O",
    )
    parser.add_argument(
        "--batch_size", type=int, default=10000, help="Batch size for queue"
    )

    args = parser.parse_args()

    # Expand paths
    input_dir = os.path.expanduser(args.input_dir)
    output_dir = os.path.expanduser(args.output_dir)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Starting engage_seq_v4 preprocessing with two-pass approach...")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Min sequence length: {args.min_seq_len}")
    logger.info(f"Max sequence length: {args.max_seq_len}")
    logger.info(f"Train split ratio: {args.train_split}")
    logger.info(f"Number of workers: {args.num_workers}")
    logger.info(f"Number of reader processes: {args.num_reader_processes}")

    # Create a Manager for shared queue
    manager = mp.Manager()

    # =================================================================
    # PASS 1: Collect all unique ad_tokens from all workers
    # =================================================================
    logger.info(f"\n{'='*60}")
    logger.info("PASS 1: Collecting unique ad_tokens from all data")
    logger.info(f"{'='*60}")

    # Create temporary directory for pass 1 outputs
    pass1_dir = os.path.join(output_dir, "pass1_tokens")
    os.makedirs(pass1_dir, exist_ok=True)

    # Create queue for pass 1
    in_q_pass1 = manager.Queue(maxsize=args.num_reader_processes * 8)

    # Start worker processes for pass 1
    workers_pass1: List[mp.Process] = []
    for worker_id in range(args.num_workers):
        worker_token_file = os.path.join(pass1_dir, f"worker_{worker_id}_tokens.txt")
        p = mp.Process(
            target=worker_loop_pass1,
            args=(
                in_q_pass1,
                worker_token_file,
                worker_id,
            ),
            daemon=True,
        )
        p.start()
        workers_pass1.append(p)

    # Read all data (both train and eval) for pass 1
    iter_json_batches_from_folder(
        input_dir,
        args.batch_size,
        in_q_pass1,
        split="train",  # Use "train" but read full dataset
        train_split_ratio=1.0,  # Read 100% of files in pass 1
        num_reader_processes=args.num_reader_processes,
    )

    logger.info("Pass 1 producer completed")

    # Signal workers to finish
    for _ in workers_pass1:
        in_q_pass1.put(SENTINEL)

    # Join workers
    for p in workers_pass1:
        p.join()

    logger.info("Pass 1: All workers completed")

    # Merge all tokens from workers and build global ad_token_to_id mapping
    logger.info("Pass 1: Merging tokens from all workers...")
    all_ad_tokens: Set[str] = set()
    for worker_id in range(args.num_workers):
        worker_token_file = os.path.join(pass1_dir, f"worker_{worker_id}_tokens.txt")
        if os.path.exists(worker_token_file):
            with open(worker_token_file, "r") as f:
                for line in f:
                    all_ad_tokens.add(line.strip())

    logger.info(f"Pass 1: Collected {len(all_ad_tokens)} unique ad_tokens")

    # Create global mapping: ad_token -> id
    ad_token_to_id: Dict[str, int] = {}
    for idx, token in enumerate(sorted(all_ad_tokens)):
        ad_token_to_id[token] = idx

    # Save the mapping to disk
    vocab_file = os.path.join(output_dir, "ad_token_vocab.json")
    with open(vocab_file, "w") as f:
        json.dump(ad_token_to_id, f)

    logger.info(f"Pass 1: Saved ad_token vocabulary to {vocab_file}")
    logger.info(f"Pass 1: Vocabulary size: {len(ad_token_to_id)}")

    # =================================================================
    # PASS 2: Process data with global ad_token_to_id mapping
    # =================================================================
    logger.info(f"\n{'='*60}")
    logger.info("PASS 2: Processing data with global vocabulary")
    logger.info(f"{'='*60}")

    all_stats = {}

    for split in ["train", "eval"]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Pass 2: Processing {split} split...")
        logger.info(f"{'='*60}")

        # Create split directory for worker outputs
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        # Create queue for pass 2
        in_q_pass2 = manager.Queue(maxsize=args.num_reader_processes * 8)

        # Start worker processes for pass 2
        workers_pass2: List[mp.Process] = []
        for worker_id in range(args.num_workers):
            worker_output_file = os.path.join(split_dir, f"worker_{worker_id}.jsonl")
            p = mp.Process(
                target=worker_loop_pass2,
                args=(
                    in_q_pass2,
                    worker_output_file,
                    args.min_seq_len,
                    args.max_seq_len or 999999,
                    worker_id,
                    ad_token_to_id,
                ),
                daemon=True,
            )
            p.start()
            workers_pass2.append(p)

        # Start producer (multi-process file readers)
        iter_json_batches_from_folder(
            input_dir,
            args.batch_size,
            in_q_pass2,
            split,
            args.train_split,
            args.num_reader_processes,
        )

        logger.info(f"Pass 2: Producer completed for {split} split")

        # Signal workers to finish
        for _ in workers_pass2:
            in_q_pass2.put(SENTINEL)

        # Join workers
        for p in workers_pass2:
            p.join()

        logger.info(f"Pass 2: All workers completed for {split} split")

        # Collect statistics and merge files
        stats = collect_statistics(output_dir, split, args.num_workers)
        all_stats[split] = stats

    # Save metadata
    metadata = {
        "num_train_users": all_stats["train"]["total_users"],
        "num_eval_users": all_stats["eval"]["total_users"],
        "num_unique_ads": max(
            all_stats["train"]["unique_ad_ids"], all_stats["eval"]["unique_ad_ids"]
        ),
        "max_ad_id": len(ad_token_to_id) - 1,
        "new_max_id": len(ad_token_to_id) - 1,
        "num_mapped_ads": len(ad_token_to_id),
        "min_sequence_length": args.min_seq_len,
        "max_sequence_length": args.max_seq_len,
        "train_split_ratio": args.train_split,
    }

    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("Preprocessing completed successfully!")
    logger.info(f"{'='*60}")
    logger.info(f"Train users: {metadata['num_train_users']}")
    logger.info(f"Eval users: {metadata['num_eval_users']}")
    logger.info(f"Unique ads: {metadata['num_unique_ads']}")
    logger.info(f"Max ad ID: {metadata['max_ad_id']}")
    logger.info(f"New max ID (mapped): {metadata['new_max_id']}")
    logger.info(f"Number of mapped ads: {metadata['num_mapped_ads']}")
    logger.info(f"Metadata saved to: {metadata_file}")
    logger.info(f"Vocabulary saved to: {vocab_file}")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    # Make Ctrl+C kill children too
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    main()
