import os
import datasets
import jsonlines
import argparse
import random

random.seed(42)


# 列出目录内容的函数
def list_directory_contents(dir_path):
    print(f"Listing contents of directory: {dir_path}")
    try:
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            contents = os.listdir(dir_path)
            if contents:
                print("Files and directories found:")
                for item in contents:
                    print(f"- {item}")
            else:
                print("The directory is empty.")
        else:
            print(f"Error: Directory not found or is not a valid directory: {dir_path}")
    except Exception as e:
        print(f"An error occurred while listing directory contents: {e}")
    print("-" * 20)  # 分隔符


def process_subset(data_path, subset_name=None):
    """处理单个子集数据文件"""
    print(f"Processing data file: {data_path}")

    # 检查文件是否存在
    if not os.path.exists(data_path):
        print(f"Warning: File not found at {data_path}")
        return None

    # 读取数据
    lines = []
    try:
        with jsonlines.open(data_path) as reader:
            for line in reader:
                lines.append(line)
    except Exception as e:
        print(f"Error reading file {data_path}: {e}")
        return None

    # 处理数据，根据 LongBench 数据格式
    all_data = []
    for line in lines:
        # LongBench 数据格式包含 'input', 'context', 'answers', 'length', 'dataset', 'language', 'all_classes', '_id' 字段
        processed_item = {
            "data_source": "longbench",
            "question": line.get("input", ""),
            "ability": "qa",
            "reward_model": {"style": "rule", "ground_truth": line.get("answers", [])},
            "golden_answers": line.get("answers", []),
            "extra_info": {
                "id": line.get("_id", ""),
                # "dataset": line.get("dataset", subset_name or "unknown"),
                # "length": line.get("length", 0),
                # "language": line.get("language", "en"),
                # "context": line.get("context", ""),
                # "all_classes": line.get("all_classes", None),
            },
        }
        if "musique" in data_path:
            print(f"Processed item: {processed_item}")  # Debugging output
        all_data.append(processed_item)

    print(f"Processed {len(all_data)} examples from {os.path.basename(data_path)}")
    return all_data


def save_dataset(data, output_path):
    """保存数据集到指定路径"""
    if not data or len(data) == 0:
        print(f"No data to save for {output_path}")
        return

    dataset = datasets.Dataset.from_list(data)
    print(f"Saving dataset ({len(data)} examples) to: {output_path}")
    dataset.to_parquet(output_path)


if __name__ == "__main__":
    # 列出目录的内容以查看可用的数据文件
    longbench_data_dir = "/home/ps/Projects/ReSearch-1/LongBench_1744473763/data"
    list_directory_contents(longbench_data_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_dir", default="/home/ps/Projects/ReSearch-1/LongBench_1744473763/data"
    )
    parser.add_argument(
        "--subset",
        nargs="+",
        default=None,
        help="Specify one or more subset names to process",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save output files, defaults to local_dir",
    )

    args = parser.parse_args()

    # 设置输出目录
    output_dir = args.output_dir or args.local_dir

    # 确定要处理的子集列表
    subsets_to_process = []

    if args.subset:
        # 如果指定了子集，使用指定的子集名称
        subsets_to_process = args.subset
    else:
        # 否则处理目录中所有的 .jsonl 文件作为子集
        if os.path.exists(args.local_dir):
            for filename in os.listdir(args.local_dir):
                if filename.endswith(".jsonl"):
                    # 从文件名中提取子集名称（去掉.jsonl扩展名）
                    subset_name = os.path.splitext(filename)[0]
                    subsets_to_process.append(subset_name)

    if not subsets_to_process:
        print("No subsets found to process.")
        exit(1)

    print(f"Will process the following subsets: {', '.join(subsets_to_process)}")

    # 处理每个子集
    for subset in subsets_to_process:
        print(f"\n{'=' * 50}\nProcessing subset: {subset}\n{'=' * 50}")

        # 构建数据文件路径
        data_path = os.path.join(args.local_dir, f"{subset}.jsonl")

        # 处理数据
        test_data = process_subset(data_path, subset)
        if not test_data:
            print(f"Skipping subset {subset} due to errors or empty data")
            continue

        # 构建输出文件路径 - 只保存为测试集
        output_test_path = os.path.join(output_dir, f"{subset}", "test.parquet")

        # 保存数据集
        save_dataset(test_data, output_test_path)

    print("\nAll subset processing complete.")
