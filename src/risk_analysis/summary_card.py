import pandas as pd
from typing import List
from collections import Counter
from src.risk_detection.risk_score_calc import monthly_risk_calc
from src.risk_detection.summary_reviews import summary_pipeline, str_to_list_keyword, EXCEPT_KEYWORD

