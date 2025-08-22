import re

# 정규표현식 연습
'''
중요 문법 5개
.   : 1개의 문자
[]  : 여러개 중 1개의 문자([abc] a, b, c 중 1개, [a-z] 범위 연산 가능)
?   : 패턴이 있거나 없거나 (colou? == "color", "colour" ...)
*   : 패턴이 0번 이상 (ab*c == "ac", "abc", "abbc" ...)
+   : 패턴이 1번 이상 (ab+c == "abc", "abbc", "abbbc" ...)
'''

db = '''3412    [BOB] 123
3834  Jonny 333
1248   Kate 634
1423   Tony 567
2567  Peter 435
3567  Alice 535
1548  Kerry 534'''
#
# # print(re.findall(r'[0123456789]', db))
# print(re.findall(r'[0-9]', db))
# print(re.findall(r'[0-9]+', db)) # + 추가로 토큰 단위로 검색
#
# # 퀴즈 : 이름을 찾아보세요
# print(re.findall(r'[A-z]+', db)) # ['[Bob]', 'Jonny', 'Kate', 'Tony', 'Peter', 'Alice', 'Kerry']
# print(re.findall(r'[A-Za-z]+', db)) # [] 를 제외한 이름 찾기
# print(re.findall(r'[A-Z|a-z]+', db)) # 틀린 표현
#
# print(re.findall(r'[A-Z][a-z]+', db)) # camel case 찾기

# ---------------------------------------------------------#

import requests

# http://openhangul.com/nlp_ko2en?q=비밀번호
response = requests.get('http://openhangul.com/nlp_ko2en?q=비밀번호')
# print(response)
# print(response.text)

# text = response.content.decode('utf-8')
# print(text)
#
# result = re.findall(r'<img src="images/cursor.gif"><br>(.+)', text) # () : 찾는것만 반환
# print(result)

response = requests.get('https://lib.yongin.go.kr/seatmate_hd/seatmate.php?classInfo=2')
print(response)
print(response.text)

text = response.content.decode('utf-8')
print(text)

result = re.findall(r'배정(.+)', text) # () : 찾는것만 반환
print(result)
