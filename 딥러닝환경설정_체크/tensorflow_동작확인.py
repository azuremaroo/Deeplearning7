# ================= 안내 메시지 제거 시작 =================
# 출력 화면에 빨간색 글씨는 tensorflow GPU 사용 여부 안내 메시지 임.(오류 아님)
# 안내 메시지 제거시 ==> 다음 처럼 코드 수정
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 최소 경고 수준으로 설정
import tensorflow as tf
# ================= 안내 메시지 제거 끝 =================

import numpy as np
import pandas as pd
import tensorflow as tf
a = tf.constant([1,2,3,4])
print(a)  # 결과 출력되면 성공

