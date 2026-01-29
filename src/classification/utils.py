import random
import numpy as np
import pandas as pd
import torch


# 전역 시드 설정
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 클래스 균등 추출
def balanced_class_extract(df: pd.DataFrame, label: str, num: int, seed: int | None = None) -> pd.DataFrame:
    balanced = (
        df.groupby(label, group_keys=False)
          .apply(lambda g: g.sample(n=num, random_state=seed), include_groups=False)
    )

    return balanced.sample(frac=1, random_state=seed).reset_index(drop=True) # 결과들을 다시 섞은 후 0부터 인덱스 정렬