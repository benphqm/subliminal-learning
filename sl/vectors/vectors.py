from dataclasses import dataclass, asdict
import torch

@dataclass
class ApproxVectorResult: 
    """HiddenStateResult, averaged over many vectors in some dimension"""
    vector: torch.Tensor
    concentration: torch.Tensor
    n_samples: int

@dataclass
class TeacherVector(ApproxVectorResult):
    s_trait: str
    s_neutral: str

@dataclass
class StudentVector(ApproxVectorResult):
    generation_run: str
    finetuning_run: str
    rank: str
    step: str

def save_vectors(path, vectors: dict[str, ApproxVectorResult]):
    serialized = {}
    for name, vector in vectors.items():
        data = asdict(vector)
        data["__type__"] = type(vector).__name__
        serialized[name] = data
    torch.save(serialized, path)



def load_vectors(path):
    TYPE_MAP = {
    "TeacherVector": TeacherVector,
    "StudentVector": StudentVector,
    "ApproxVectorResult": ApproxVectorResult,
}
    raw = torch.load(path, weights_only=False)
    vectors = {}
    for name, data in raw.items():
        cls = TYPE_MAP.get(data.pop("__type__"), ApproxVectorResult)
        vectors[name] = cls(**data)

    return vectors