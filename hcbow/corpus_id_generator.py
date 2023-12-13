import json
import os
from konlpy.tag import Komoran
import pickle

# 에세이를 읽어오는 함수
def read_essays(DIR_PATH):
    essays = []
    files = [f for f in os.listdir(DIR_PATH) if f.endswith('.json')]
    for file_name in files:
        file_path = os.path.join(DIR_PATH, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            # essays.append(data['passage'])
            essays.append(data['paragraph'][0]['paragraph_txt'])
    return essays

# 에세이 전처리 함수
def preprocess_essay(essay):
    essay = essay.replace('\n', '')
    essay = essay.encode('utf-8', 'ignore').decode('utf-8')
    return essay

# 형태소 분석 및 고유 ID 할당 함수
def vocabulary(text, corpus=None, word_to_id=None, id_to_word=None):
    if word_to_id is None:
        word_to_id = {}
    if id_to_word is None:
        id_to_word = {}
    if corpus is None:
        corpus = []

    komoran = Komoran()
    words = komoran.nouns(text)
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
        corpus.append(word_to_id[word])

    return corpus, word_to_id, id_to_word

# Pickle 파일을 로드하는 함수
def load_pickle(file_path):
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        return None

# Pickle 파일을 저장하는 함수
def save_pickle(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

#################################################################################
        
# 기존 데이터 로드
global_corpus = load_pickle('book_corpus.pkl') or []
global_word_to_id = load_pickle('book_word_to_id.pkl') or {}
global_id_to_word = load_pickle('book_id_to_word.pkl') or {}

directories = ['글짓기', '대안제시', '설명글', '주장', '찬성반대']

# 새 에세이 데이터 로드 및 전처리
DIR_PATH = r'C:\Users\keaveno\Desktop\Programming\jupyter\deep\dataset'.replace('\\', '/')

for d in directories:
    print(f'---{d} 시작---')
    dir_path = os.path.join(DIR_PATH, d)
    new_essays = read_essays(dir_path)
    i = 1
    for essay in new_essays:
        try:
            text = preprocess_essay(essay)
            global_corpus, global_word_to_id, global_id_to_word = vocabulary(text, global_corpus, global_word_to_id, global_id_to_word)
            if i%100 == 0:
                print(f'완료 {i}')
        except:
            print(f'*** {i} 실패 ***')
        i += 1
        
    # 업데이트된 데이터 저장
    save_pickle(global_corpus, 'added_book_corpus.pkl')
    save_pickle(global_word_to_id, 'added_book_word_to_id.pkl')
    save_pickle(global_id_to_word, 'added_book_id_to_word.pkl')
    print('데이터 저장 완료')
    
    print(f'\n---{d} 완료---\n')

print('완료')


directories = ['예술', '사회과학', '기타', '기술과학']

print(f'-------------Valid 시작-------------\n') #################

# 새 에세이 데이터 로드 및 전처리
DIR_PATH = r'C:\Users\keaveno\Downloads\도서자료 요약\Validation\[원천]도서요약_valid'.replace('\\', '/')

for d in directories:
    print(f'---{d} 시작---')
    dir_path = os.path.join(DIR_PATH, d)
    new_essays = read_essays(dir_path)
    i = 1
    for essay in new_essays:
        try:
            text = preprocess_essay(essay)
            global_corpus, global_word_to_id, global_id_to_word = vocabulary(text, global_corpus, global_word_to_id, global_id_to_word)
            if i%100 == 0:
                print(f'완료 {i}')
        except:
            print(f'*** {i} 실패 ***')
        i += 1
        
    # 업데이트된 데이터 저장
    save_pickle(global_corpus, 'corpus.pkl')
    save_pickle(global_word_to_id, 'word_to_id.pkl')
    save_pickle(global_id_to_word, 'id_to_word.pkl')
    print('데이터 저장 완료')
    
    print(f'\n---{d} 완료---\n')

print(f'\n-------------Valid 완료-------------\n') ###

print(f'-------------Train 시작-------------\n') ######################

# 새 에세이 데이터 로드 및 전처리
DIR_PATH = r'C:\Users\keaveno\Downloads\도서자료 요약\Training\[원천]도서요약_train'.replace('\\', '/')

for d in directories:
    print(f'---{d} 시작---')
    dir_path = os.path.join(DIR_PATH, d)
    new_essays = read_essays(dir_path)
    i = 1
    for essay in new_essays:
        try:
            text = preprocess_essay(essay)
            global_corpus, global_word_to_id, global_id_to_word = vocabulary(text, global_corpus, global_word_to_id, global_id_to_word)
            if i%100 == 0:
                print(f'완료 {i}')
        except:
            print(f'*** {i} 실패 ***')
        i += 1
        
    # 업데이트된 데이터 저장
    save_pickle(global_corpus, 'corpus.pkl')
    save_pickle(global_word_to_id, 'word_to_id.pkl')
    save_pickle(global_id_to_word, 'id_to_word.pkl')
    print('데이터 저장 완료')
    
    print(f'\n---{d} 완료---\n')

print(f'\n-------------Train 완료-------------\n') ###
