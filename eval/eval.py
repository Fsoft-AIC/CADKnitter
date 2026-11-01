import argparse
import pandas as pd
import os
import json
import pickle

from evaluators import ContactEvaluation, ChamferEvaluation, CollisionEvaluation
from tqdm import tqdm

def get_object_name(object_filename):
    return object_filename.split('/')[-1].split('.')[0].replace('merged_', '').replace('output_', '')

def load_condition_contact_surface_ids(joint_set_dir, object_name):
    joint_set_file = os.path.join(joint_set_dir, f'{object_name}.pkl')
    
    if not os.path.exists(joint_set_file):
        raise FileNotFoundError(f"Joint set file not found: {joint_set_file}")
    
    try:
        with open(joint_set_file, 'rb') as f:
            joint_data = pickle.load(f)
        
        if 'contact_surfaces' in joint_data:
            cond_contact_surface_ids = set()
            for contact_pair in joint_data['contact_surfaces']:
                # @TODO: Hardcoded index. Current evaluation protocol uses the second object as the condition object.
                cond_contact_surface_ids.update(contact_pair['surface2_id'])
            cond_contact_surface_ids = list(cond_contact_surface_ids)
            return cond_contact_surface_ids
        else:
            raise ValueError(f"No contact surfaces found in {joint_set_file}")
            
    except Exception as e:
        raise ValueError(f"Error loading joint set file {joint_set_file}: {e}")

def main(args):
    # Initialize results dataframe
    metrics = args.metrics.split(',')
    metric_names = []
    for metric in metrics:
        if metric == 'contact':
            metric_names.extend(ContactEvaluation.get_metric_names())
        elif metric == 'chamfer':
            metric_names.extend(ChamferEvaluation.get_metric_names())
        elif metric == 'collision':
            metric_names.extend(CollisionEvaluation.get_metric_names())
        else:
            raise ValueError(f"Invalid metric: {metric}")

    print(f"Evaluating {len(metric_names)} metrics: {metric_names}")
    identifier_columns = ['object_idx', 'sample_id', 'object_name']
    pd_columns = identifier_columns + metric_names
    # Prepare output directory
    output_subdir = args.generated_dir.split('/')[-1]
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, output_subdir), exist_ok=True)
    results_csv_path = os.path.join(args.output_dir, output_subdir, 'evaluation_results.csv')
    grouped_csv_path = os.path.join(args.output_dir, output_subdir, 'evaluation_grouped.csv')
    summary_json_path = os.path.join(args.output_dir, output_subdir, 'evaluation_summary.json')
    evaluated_samples_path = os.path.join(args.output_dir, output_subdir, 'evaluated_samples.txt')
    
    # Load test list
    if os.path.isfile(args.test_list):
        if args.test_list.endswith('.txt'):
            with open(args.test_list, 'r') as f:
                test_list = [line.strip() for line in f.readlines()]
        elif args.test_list.endswith('.json'):
            with open(args.test_list, 'r') as f:
                test_list = json.load(f)
                test_list = test_list[args.test_split]
        else:
            raise ValueError(f"Invalid test list file: {args.test_list}")
        
        end_idx = len(test_list) if args.end_idx is None else args.end_idx
        test_list = test_list[args.start_idx:end_idx]
    else:
        raise ValueError(f"Invalid test list file: {args.test_list}")
    
    print(f"Evaluating {len(test_list)} test samples")
    
    # Load the existing results dataframe
    if os.path.isfile(results_csv_path) and not args.override:
        results_df = pd.read_csv(results_csv_path, dtype={'object_name': str})
    else:
        results_df = pd.DataFrame(columns=pd_columns)

    # Skip already evaluated samples
    if os.path.isfile(evaluated_samples_path) and not args.override:
        with open(evaluated_samples_path, 'r') as f:
            evaluated_samples = [line.strip() for line in f.readlines()]
        test_list = [x for x in test_list if x not in evaluated_samples]
    else:
        evaluated_samples = []

    for object_idx, object_filename in tqdm(enumerate(test_list), total=len(test_list)):
        object_name = get_object_name(object_filename)
        condition_step_file = f'{args.ground_truth_dir}/{object_name}_cond.step'
        ground_truth_step_file = f'{args.ground_truth_dir}/{object_name}_gt.step'
        condition_stl_file = f'{args.ground_truth_dir}/{object_name}_cond.stl'
        ground_truth_stl_file = f'{args.ground_truth_dir}/{object_name}_gt.stl'

        if ('contact' in metrics and not os.path.isfile(condition_step_file)) or not os.path.isfile(ground_truth_step_file) \
            or not os.path.isfile(condition_stl_file):
            continue

        if 'contact' in metrics:
            condition_contact_surface_ids = load_condition_contact_surface_ids(args.joint_set_pkl_dir, object_name)

        for sample_idx in range(args.num_samples_per_object):
            new_row = {
                'object_idx': object_idx,
                'object_name': object_name,
                'sample_id': sample_idx,
            }
            generated_step_file = f'{args.generated_dir}/output_{object_name}_{sample_idx}.step'
            generated_stl_file = f'{args.generated_dir}/output_{object_name}_{sample_idx}.stl'
            if ('contact' in metrics and not os.path.isfile(generated_step_file)) or not os.path.isfile(generated_stl_file) or not os.path.isfile(ground_truth_stl_file):
                continue
            
            for metric in metrics:
                scores = None
                # print(f"Evaluating {metric}")
                if metric == 'contact':
                    eval_contact = ContactEvaluation(
                        condition_step_file=condition_step_file,
                        generated_step_file=generated_step_file,
                        condition_contact_surface_ids=condition_contact_surface_ids,
                        tolerance=args.tolerance,
                        num_u=args.num_u,
                        num_v=args.num_v,
                        threshold=args.threshold,
                        args=args
                    )
                    scores = eval_contact.evaluate()
                elif metric == 'chamfer':
                    eval_chamfer = ChamferEvaluation(
                        ground_truth_stl_file=ground_truth_stl_file,
                        generated_stl_file=generated_stl_file,
                        args=args
                    )
                    scores = eval_chamfer.evaluate()
                elif metric == 'collision':
                    eval_collision = CollisionEvaluation(
                        condition_stl_file=condition_stl_file,
                        generated_stl_file=generated_stl_file,
                        args=args
                    )
                    scores = eval_collision.evaluate()
                else:
                    raise ValueError(f"Invalid metric: {metric}")
                
                if scores is not None:
                    for key, value in scores.items():
                        new_row[key] = value
                        
            # Check if new_row has enough fields to be added to dataframe
            if len(new_row) == len(pd_columns):
                # Add result to dataframe (skip concat while empty: concat with an empty frame is deprecated in pandas)
                new_row_df = pd.DataFrame([new_row], columns=pd_columns)
                results_df = new_row_df if results_df.empty else pd.concat([results_df, new_row_df], ignore_index=True)

        # Write result
        results_df.to_csv(results_csv_path, index=False)
        evaluated_samples.append(object_filename)
        with open(evaluated_samples_path, 'w') as f:
            f.write("\n".join(evaluated_samples))

    # Save detailed results as CSV
    print(f"Detailed results saved to: {results_csv_path}")
    
    summary_dict = {}
    results_df = results_df.drop(columns=['object_idx', 'sample_id'])
    # Report mean of each metric group by the object name
    results_df_grouped = results_df.groupby('object_name').mean()
    results_df_grouped.to_csv(grouped_csv_path, index=False)

    # Report the mean of each group
    results_df_grouped_mean = results_df_grouped.mean()
    results_df_grouped_mean_dict = results_df_grouped_mean.to_dict()

    # Report the average valid ratio over all evaluated objects, including those from previous (resumed) runs.
    # Each row in results_df is one valid sample; objects with no valid sample are absent from it and count as 0.
    valid_counts = results_df.groupby('object_name').size()
    valid_ratios = [valid_counts.get(get_object_name(x), 0) / args.num_samples_per_object for x in evaluated_samples]
    results_df_grouped_mean_dict['valid_ratio'] = sum(valid_ratios) / len(valid_ratios) if len(valid_ratios) > 0 else None
    summary_dict['grouped_mean'] = results_df_grouped_mean_dict

    with open(summary_json_path, 'w') as f:
        json.dump(summary_dict, f, indent=4)
    print(f"Summary results saved to: {summary_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generated CAD models using specified metrics")
    parser.add_argument("--generated_dir", type=str, required=True, 
                       help="Directory containing generated STEP files")
    parser.add_argument("--ground_truth_dir", type=str, required=True,
                       help="Directory containing ground truth STEP files")
    parser.add_argument("--joint_set_pkl_dir", type=str, required=True,
                       help="Directory containing joint set information")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save evaluation results")
    parser.add_argument("--test_list", type=str, required=True,
                       help="Path to file containing list of test samples")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start index")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="End index")
    parser.add_argument("--test_split", type=str, default='test',
                       help="Test split to evaluate")
    parser.add_argument("--metrics", type=str, required=True,
                       help="Comma-separated list of metrics to evaluate (e.g., 'contact,chamfer')")
    parser.add_argument("--num_samples_per_object", type=int, default=16,
                       help="Number of samples to evaluate per object")
    parser.add_argument("--tolerance", type=float, default=0.1,
                       help="Tolerance for contact evaluation")
    parser.add_argument("--num_u", type=int, default=10,
                       help="Number of U samples for face evaluation")
    parser.add_argument("--num_v", type=int, default=10,
                       help="Number of V samples for face evaluation")
    parser.add_argument("--threshold", type=float, default=0.1,
                       help="Area threshold for contact surface evaluation")
    parser.add_argument("--override", action='store_true',
                       help="Override existing results")
    args = parser.parse_args()

    main(args)