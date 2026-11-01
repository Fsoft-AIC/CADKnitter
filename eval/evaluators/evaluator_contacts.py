from OCC.Core.gp import gp_Pnt
from OCC.Core.TopoDS import TopoDS_Face
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.BRep import BRep_Tool
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex

from occwl.face import Face
from occwl.uvgrid import uvgrid
import numpy as np

from functools import partial

from occwl.compound import Compound
from occwl.entity_mapper import EntityMapper

from .evaluator import Evaluator


class ContactEvaluation(Evaluator):
    def __init__(self, 
                 condition_step_file, generated_step_file, condition_contact_surface_ids,
                 tolerance=0.1, num_u=32, num_v=32, threshold=0.1, args=None):
        super().__init__(args)
        self.tolerance = tolerance
        self.num_u = num_u
        self.num_v = num_v
        self.threshold = threshold # Area threshold of the contact surface
        self.condition_step_file = condition_step_file
        self.generated_step_file = generated_step_file
        self.condition_contact_surface_ids = condition_contact_surface_ids

    def get_metric_names():
        return ['proximity']

    def face_proximity(self, condition_shape, generated_shape, tolerance=0.1, condition_contact_checker=None):
        condition_shape_faces = []
        generated_shape_faces = []
        condition_face_indices = []
        generated_face_indices = []
        face_indices_pairs = []

        for condition_face_idx, condition_face in enumerate(condition_shape.faces()):
            if not condition_contact_checker(condition_face):
                continue
            # print(f"Checking with condition contact surface {condition_face_idx}")
            for generated_face_idx, generated_face in enumerate(generated_shape.faces()):
                try:
                    dss = BRepExtrema_DistShapeShape()
                    dss.LoadS1(condition_face.topods_shape())
                    dss.LoadS2(generated_face.topods_shape())
                    dss.Perform()
                except Exception as e:
                    print(f"Error performing distance shapes in face_proximity: {e}")
                    continue
                
                if not dss.IsDone():
                    continue
                
                if dss.Value() > tolerance:
                    continue
                
                if condition_face_idx not in condition_face_indices:
                    condition_shape_faces.append(condition_face.topods_shape())
                    condition_face_indices.append(condition_face_idx)
                if generated_face_idx not in generated_face_indices:
                    generated_shape_faces.append(generated_face.topods_shape())
                    generated_face_indices.append(generated_face_idx)

                condition_face_index = condition_face_indices.index(condition_face_idx)
                generated_face_index = generated_face_indices.index(generated_face_idx)
                face_indices_pairs.append((condition_face_index, generated_face_index))

        return condition_shape_faces, generated_shape_faces, face_indices_pairs # two lists of TopoDS_Shape + matched index pairs

    def points_to_face(self, points, face, normals, points2, normals2, tolerance=0.1):
        dist = 0
        count = 0
        for point_idx, point in enumerate(points):
            pnt = gp_Pnt(point[0], point[1], point[2])
            surface = BRep_Tool.Surface(face)
            pps = GeomAPI_ProjectPointOnSurf(pnt, surface)
            vertex = BRepBuilderAPI_MakeVertex(pnt).Vertex()
            dss = BRepExtrema_DistShapeShape()
            dss.LoadS1(vertex)
            dss.LoadS2(face)
            dss.Perform()
            
            if not dss.IsDone() or not pps.IsDone():
                continue
            
            if dss.Value() > tolerance or pps.LowerDistance() > tolerance:
                continue

            # Approximate the normal of the projected point using the normal of 4 points closest to the projected point in points2
            projected_point = np.array(pps.NearestPoint().Coord())
            closest_points2_indices = np.argsort(np.linalg.norm(points2 - projected_point, axis=-1))[:4]
            if len(closest_points2_indices) == 0:
                continue
            projected_normal = 0
            for i in range(len(closest_points2_indices)):
                projected_normal += normals2[closest_points2_indices[i]]
            projected_normal /= 4
            norm_projected_normals = np.linalg.norm(projected_normal)
            norm_normals = np.linalg.norm(normals[point_idx])
            normalized_normals = normals[point_idx] / (norm_normals)
            normalized_projected_normals = projected_normal / (norm_projected_normals)
            cosine_angle = np.dot(normalized_projected_normals, normalized_normals)

            if cosine_angle > -0.99:
                continue

            dist += pps.LowerDistance()
            count += 1
            
        return dist / (count + 1e-6), count

    def face_coincidence(self, face1: TopoDS_Face, face2: TopoDS_Face, 
                        points1: np.ndarray, points2: np.ndarray, 
                        normals1: np.ndarray, normals2: np.ndarray,
                        tolerance=0.1, threshold=0.1):
        """
        Check if two faces are coincident.
        """

        mean_dist_to_face2, count_points_to_face2 = self.points_to_face(points1, face2, normals1, points2, normals2, tolerance)
        if count_points_to_face2 / (len(points1) + 1e-6) >= threshold:
            return mean_dist_to_face2, True
                
        mean_dist_to_face1, count_points_to_face1 = self.points_to_face(points2, face1, normals2, points1, normals1, tolerance)
        if count_points_to_face1 / (len(points2) + 1e-6) >= threshold:
            return mean_dist_to_face1, True
        
        return min(mean_dist_to_face2, mean_dist_to_face1), False

    def find_face_coincidence(self, condition_shape, generated_shape, condition_contact_checker=None):
        condition_shape_faces, generated_shape_faces, face_indices_pairs = self.face_proximity(condition_shape, generated_shape, self.tolerance, condition_contact_checker)
        # Sampling points and normals from faces and ignore the boundary points of the faces
        condition_points = {face_idx: uvgrid(Face(face), num_u=self.num_u, num_v=self.num_v, method="point")[1:-1, 1:-1].reshape(-1, 3) for face_idx, face in enumerate(condition_shape_faces)}
        generated_points = {face_idx: uvgrid(Face(face), num_u=self.num_u, num_v=self.num_v, method="point")[1:-1, 1:-1].reshape(-1, 3) for face_idx, face in enumerate(generated_shape_faces)}
        condition_normals = {face_idx: uvgrid(Face(face), num_u=self.num_u, num_v=self.num_v, method="normal")[1:-1, 1:-1].reshape(-1, 3) for face_idx, face in enumerate(condition_shape_faces)}
        generated_normals = {face_idx: uvgrid(Face(face), num_u=self.num_u, num_v=self.num_v, method="normal")[1:-1, 1:-1].reshape(-1, 3) for face_idx, face in enumerate(generated_shape_faces)}
        condition_masks = {face_idx: uvgrid(Face(face), num_u=self.num_u, num_v=self.num_v, method="inside")[1:-1, 1:-1].reshape(-1) for face_idx, face in enumerate(condition_shape_faces)}
        generated_masks = {face_idx: uvgrid(Face(face), num_u=self.num_u, num_v=self.num_v, method="inside")[1:-1, 1:-1].reshape(-1) for face_idx, face in enumerate(generated_shape_faces)}
        
        for face_idx, mask in condition_masks.items():
            condition_points[face_idx] = condition_points[face_idx][mask]
            condition_normals[face_idx] = condition_normals[face_idx][mask]
        for face_idx, mask in generated_masks.items():
            generated_points[face_idx] = generated_points[face_idx][mask]
            generated_normals[face_idx] = generated_normals[face_idx][mask]

        coincident_faces = set()
        face_face_distances = []
        for condition_face_idx, condition_face in enumerate(condition_shape_faces):
            min_face_face_dist = 1e10
            has_coincidence = False 
            for generated_face_idx, generated_face in enumerate(generated_shape_faces):
                if (condition_face_idx, generated_face_idx) not in face_indices_pairs:
                    continue

                face_face_dist, is_coincidence = self.face_coincidence(condition_face, 
                                                        generated_face, 
                                                        condition_points[condition_face_idx], 
                                                        generated_points[generated_face_idx], 
                                                        condition_normals[condition_face_idx], 
                                                        generated_normals[generated_face_idx], 
                                                        self.tolerance, 
                                                        self.threshold)
                if is_coincidence:
                    has_coincidence = True
                    coincident_faces.add(condition_face)
                    if min_face_face_dist > face_face_dist:
                        min_face_face_dist = face_face_dist
            if has_coincidence:
                face_face_distances.append(min_face_face_dist)

        assert len(list(coincident_faces)) == len(face_face_distances)
                
        mean_face_dist = sum(face_face_distances) / len(face_face_distances) if len(face_face_distances) > 0 else None
        return mean_face_dist

    def is_condition_contact_surface(self, face, entity_mapper=None):
        if entity_mapper is None:
            return False
        
        face_idx = entity_mapper.face_index(face)
        return face_idx in self.condition_contact_surface_ids

    def evaluate(self):
        condition_shape = list(Compound.load_from_step(self.condition_step_file).solids())
        generated_shape = list(Compound.load_from_step(self.generated_step_file).solids())

        if len(condition_shape) != 1 or len(generated_shape) != 1:
            print(f"Shape {self.condition_step_file} or {self.generated_step_file} is not a single solid")
            return {
                'proximity': None,
            }

        condition_solid = condition_shape[0]
        generated_solid = generated_shape[0]

        condition_solid = condition_solid.split_all_closed_faces(num_splits=0)
        generated_solid = generated_solid.split_all_closed_faces(num_splits=0)

        condition_entity_mapper = EntityMapper(condition_solid)
        condition_contact_checker = partial(self.is_condition_contact_surface, entity_mapper=condition_entity_mapper)

        mean_face_dist = self.find_face_coincidence(condition_solid, generated_solid, condition_contact_checker)

        return {
            'proximity': mean_face_dist * 10 if mean_face_dist is not None else None # reported x10 in the paper; None when no contact face coincides
        }