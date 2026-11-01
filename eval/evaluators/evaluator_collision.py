import numpy as np
import trimesh
from .evaluator import Evaluator
import math

class CollisionEvaluation(Evaluator):
    def __init__(self, condition_stl_file, generated_stl_file, args=None):
        super().__init__(args)
        self.condition_stl_file = condition_stl_file
        self.generated_stl_file = generated_stl_file
    
    def get_metric_names():
        return ['intersection_volume_percentage']
    
    def evaluate(self):
        mesh_condition = trimesh.load(self.condition_stl_file)
        mesh_generated = trimesh.load(self.generated_stl_file)

        # Scale both mesh by the maximum dim of condition mesh
        max_dim = np.max(mesh_condition.extents)
        mesh_condition.apply_scale(1.0/max_dim)
        mesh_generated.apply_scale(1.0/max_dim)
        
        if not mesh_generated.is_watertight:
            print(f"{self.generated_stl_file} is non-watertight")
            return None
        
        if not mesh_condition.is_watertight:
            print(f"{self.condition_stl_file} is non-watertight")
            return {
                'intersection_volume_percentage': None,
            }
            
        try:
            intersection_mesh = trimesh.boolean.boolean_manifold([mesh_condition, mesh_generated], 'intersection')
            union_mesh = trimesh.boolean.boolean_manifold([mesh_condition, mesh_generated], 'union')
        except ValueError as e:
            return {
                'intersection_volume_percentage': None,
            }
        
        if not intersection_mesh.is_watertight or not union_mesh.is_watertight or math.isnan(intersection_mesh.volume) or math.isnan(union_mesh.volume):
            return {
                'intersection_volume_percentage': None,
            }

        intersection_volume_percentage = intersection_mesh.volume / mesh_generated.volume * 100
        return {
            'intersection_volume_percentage': intersection_volume_percentage,
        }