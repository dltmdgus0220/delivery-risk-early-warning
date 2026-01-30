# 🛵 배달앱 리뷰 분석을 통한 이탈 리스크 감지 시스템
## “배달의민족” 고객 이탈 방지를 위한 리뷰 분석 프로젝트

## 1. 프로젝트 개요

본 프로젝트는 배달앱 리뷰 데이터 분석을 통해 고객의 이탈 위험 요인을 분석하고, 사전 예측하여 대비하기 위한 **이탈 리스크 조기 경보 시스템**입니다.

관리자가 이탈 리스크를 빠르게 인지하고, 문제 원인에 따른 즉각적인 대응이 가능하도록 지원하는 것을 목표로 합니다.



## 2. 주요 기능



## 3. 기술 스택



## 4. 데이터




## 5. 설치 및 실행
### 5.1. 환경 설정

1.  **Python 설치:** Python 3.8 이상 버전이 설치되어 있어야 합니다.
2.  **가상 환경 생성 및 활성화 (권장):**
    해당 프로젝트는 conda 가상환경을 사용했습니다.
    # Windows
    ```bash
    conda create -n env01 python=3.10
    conda activate env01
    ```
    # macOS/Linux
    ```
3.  **의존성 설치:** `requirements.txt`에 명시된 라이브러리를 설치합니다.
    ```bash
    pip install -r requirements.txt
    ```


## 6. 프로젝트 구조

```
.
├── .gitignore
├── README.md                  
├── requirements.txt # Python 의존성 목록
├── data/ # 원시/결과 데이터 저장 경로
├── model_out/ # 학습된 모델 가중치 저장 경로
├── data_collect.py # 데이터수집
└── src/
    ├── classification/
    |   ├── classifier.py # 이탈의도분류
    |   ├── configs.py
    |   ├── datasets.py
    |   ├── trainer.py
    |   └── utils.py    
    ├── keyword/
    |   ├── llm_keyword_async.py # 비동기식 키워드도출
    |   ├── llm_keyword.py # 동기식 키워드도출
    |   └── 기타파일들
    └── labeling/
        ├── llm_churn_intent_labeling_async.py # 비동기식 이탈의도라벨링
        ├── llm_churn_intent_labeling.py # 동기식 이탈의도라벨링
        └── prompt_dev/ # 프롬프트 개선을 위해 사용한 파일들


```

## 7. 평가

## 기타. 일정관리
https://www.notion.so/2d8864f0902080f79836f0036fddc088

