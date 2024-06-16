from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  # Adjusted import for langchain_openai
from openai import OpenAI  # Adjusted import for openai
import os
from langchain.document_loaders import WebBaseLoader
import tiktoken  # Assuming this is a custom module not listed in requirements.txt
from langchain.vectorstores import Chroma  # Adjusted import for langchain.vectorstores
from langchain_openai import OpenAIEmbeddings  # Adjusted import for langchain_openai
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Adjusted import for langchain.text_splitter
from langchain.chat_models import ChatOpenAI  # Adjusted import for langchain.chat_models
from langchain.chains import RetrievalQA  # Adjusted import for langchain.chains
import uvicorn



# Load API KEY information
load_dotenv()

# Initialize ChatOpenAI and OpenAI
llm = ChatOpenAI(temperature=0.1, model_name='gpt-4o')
image_classifier = OpenAI()

app = FastAPI()

class ImageData(BaseModel):
    imageUrl: str
    region: str

@app.post("/imageregion")
def process_image_data(data: ImageData):
    try:
        print(f"Received Image URL: {data.imageUrl}")
        print(f"Received Region: {data.region}")
        print("데이터 수신 완료")
        
        # 이미지 분류 로직
        print("분석중...")
        response = image_classifier.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": '''
                         너는 사진을 보면 그 물체가 어떤 재활용품인지 알 수 있는 대한민국의 분리배출 전문가야. 나는 물체가 어떤 재활용품인지 잘 몰라서 분리배출을 잘 못하는 일반인이야. 너의 답변을 통해 일반인이 올바른 분리배출을 할 수 있도록 정확한 답변을 해줘.
        소개나 인사말 등 답변과 관계 없는 말은 하지 마.
        이 사진속 물체가 무엇인지, 상태는 어떤지 확인해서 반드시 목록 중 하나로 선택해서 대답해줘.
        ###
        물체 목록 : 더러운 플라스틱, 깨끗한 플라스틱, 더러운 스티로폼, 깨끗한 스티로폼, 더러운 유색 페트병, 깨끗한 유색 페트병, 더러운 유리병, 깨진 유리병, 종이, 깨끗한 비닐, 더러운 비닐, 금속캔, 깨끗한 투명 페트병, 더러운 투명 페트병
        ###

        인식한 물체와 비슷한 항목이 물체 목록에 없는 경우에는 반드시 예외 답변 예시에 맞게 답변해줘.
        ### 
        예외 답변 양식: '인식한 물체명' 재활용불가
        ###
                        '''},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data.imageUrl,
                            },
                        },
                    ],
                }
            ],
            max_tokens=1000,
        )
        
        인식결과 = response.choices[0].message.content
        print(f"분석 결과: {인식결과}")
        

        # 여기에 적용대상 코드를 알맞게 구현해주세요.
        
        # 지역별 분리배출 안내페이지
        loc_doc = {'강동구': ('https://www.gangdong.go.kr/web/newportal/contents/gdp_005_004_010_001', 'https://www.gangdong.go.kr/web/newportal/contents/gdp_005_004_010_003_001', 'https://www.gangdong.go.kr/web/newportal/contents/gdp_005_004_010_004_001', 'https://www.gangdong.go.kr/web/newportal/contents/gdp_005_004_010_004_002', 'https://www.gangdong.go.kr/web/newportal/contents/gdp_005_004_010_004_003'), '송파구': ('https://www.songpa.go.kr/www/contents.do?key=3153', 'https://www.songpa.go.kr/www/contents.do?key=3157&', 'https://www.songpa.go.kr/www/contents.do?key=3161&', 'https://www.songpa.go.kr/www/contents.do?key=2117&', 'https://www.songpa.go.kr/www/contents.do?key=3164&', 'https://www.songpa.go.kr/www/contents.do?key=3171&'), '강서구': ('https://www.gangseo.seoul.kr/env/env010101', 'https://www.gangseo.seoul.kr/env/env010202', 'https://www.gangseo.seoul.kr/env/env010203', 'https://www.gangseo.seoul.kr/env/env010401')}

        # RAG
        # 지역별 분리배출 안내 페이지 중 사용자가 선택한 지역의 페이지를 로드

        
        loc = data.region
        data = ''

        if loc in loc_doc.keys():
            print(loc_doc[loc])
            loader = WebBaseLoader(loc_doc[loc])
            data = loader.load()
            data
        else:
            pass

        print("1번 구역 완료")

        # 페이지 내부 내용(Text)를 사이즈에 맞게 스플릿 후 저장
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
        texts = text_splitter.split_documents(data)
        print("2번 구역 완료")

        # 페이지 내부 내용을 벡터스토어에 저장
        persist_directory = f"./storage/{loc}"
        print("3.1번 구역 완료")
        embeddings = OpenAIEmbeddings()
        print("3.2번 구역 완료")
        vectordb = Chroma.from_documents(documents=texts,
                                         embedding=embeddings,
                                         persist_directory=persist_directory)
        print("3.3번 구역 완료")
        vectordb.persist()
        print("3번 전체 구역 완료")

        # 벡터스토어 내용을 검색에 활용
        retriever = vectordb.as_retriever()
        llm = ChatOpenAI(temperature=0.1,
                         max_tokens=2048,
                         model_name='gpt-4o',
                         )
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        print("4번 구역 완료")

        if "재활용불가" in 인식결과:
            print(f"해당 제품은 {인식결과[:인식결과.find(" ")]}이네요! 재활용이 불가합니다. 일반쓰레기로 불출해주세요.")  # 이미지 분류에서 재활용 불가로 나오는 경우 처리
        else:
            query = f'''
            ###Prompt 너는 대한민국의 분리배출 전문가야 나는 분리배출 방법을 잘 모르는 일반인이야. 일반인이 분리배출을 올바르게 할 수 있도록 답변을 작성해줘. 소개나 인사말, 끝인사 등 답변과 관계 없는 말은 절대로 하지 마.
            # {loc}에서 {인식결과} 분리배출 방법을 설명해줘. 반드시 답변 양식에 맞게 답해주고, 내용은 물품명과 분리배출 방법에 따라 수정해서 작성해줘. 

            ###
            답변 양식
            안녕하세요.
            작은 실천을 통해 환경을 지키는 WasteWise입니다.
            
            오늘의 분리배출 지역은 {loc}이시군요.

            해당 제품은 {인식결과}이네요! + 배출 방법을 안내해드릴게요.

            {인식결과.split()[-1]}은
            - 스티로폼에 붙어있는 음식물 찌꺼기나 이물질을 깨끗이 제거해주세요.
            - 스티로폼을 다른 재활용품과 섞이지 않도록 분리하여 배출해주세요.
            - 스티로폼을 가능한 한 작게 부수어 부피를 줄여주세요.
            ###
            '''
            try:
                llm_response = qa(query)
                print(llm_response["result"])
                
                
            except Exception as err:
                print('Exception occurred. Please try again', str(err))
        print("5번 구역 완료")



        # 처리된 데이터를 스프링 부트 서버로 전송
        print("해당 데이터를 SpringBoot로 보냅니다.")
        spring_boot_url = "http://127.0.0.1:8080/processed"
        print("6번 구역 완료")
        processed_data = {
            "imageUrl": data.imageUrl,
            "region": data.region,
            "classification": 인식결과
            # "llmResult": llm_response
        }
        print("7번 구역 완료")
        response = requests.post(spring_boot_url, json=processed_data)
        print("8번 구역 완료")
        response.raise_for_status()
        print("9번 구역 완료")
        return {"message": "Data received and sent to Spring Boot successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
print("10번 구역 완료")
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
