import pandas as pd
from google import genai
from typing import Tuple, List, Optional
import argparse
import time
import re
from collections import Counter
import os


EXCEPT_KEYWORD = ['앱-삭제', '앱-탈퇴']
