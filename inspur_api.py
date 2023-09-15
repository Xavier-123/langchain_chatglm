import argparse
import copy
import json
import logging
import os
import shutil
from typing import List, Optional, Dict
import urllib
import requests
from datetime import datetime
import uuid
import pysolr
from fastapi.responses import JSONResponse
import nltk
import pydantic
import uvicorn
from fastapi import Body, FastAPI, File, Form, Query, UploadFile, WebSocket, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing_extensions import Annotated
from starlette.responses import RedirectResponse
from typing import Dict, Any
from chains.local_doc_qa import LocalDocQA
from configs.model_config import (VS_ROOT_PATH, UPLOAD_ROOT_PATH, EMBEDDING_DEVICE,
                                  EMBEDDING_MODEL, NLTK_DATA_PATH,
                                  VECTOR_SEARCH_TOP_K, LLM_HISTORY_LEN, OPEN_CROSS_DOMAIN, VECTOR_SEARCH_SCORE_THRESHOLD)
import models.shared as shared
from models.loader.args import parser
from models.loader import LoaderCheckPoint

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path
app = FastAPI()
#solr_url = "192.168.12.84"
#solr_port = 9003


os.environ.setdefault("SOLR_FASTAPI_IP", "192.168.12.84")
os.environ.setdefault("SOLR_FASTAPI_PORT", "9003")

os.environ.setdefault("SOLR_SERVICE_IP", "192.168.12.84")
os.environ.setdefault("SOLR_SERVICE_PORT", "8983")


os.environ.setdefault("CHATGLM_IP", "0.0.0.0")
os.environ.setdefault("CHATGLM_PORT", "18015")

solr_fastapi_url = os.environ.get("SOLR_FASTAPI_IP")
solr_fastapi_port = os.environ.get("SOLR_FASTAPI_PORT")

solr_service_url = os.environ.get("SOLR_SERVICE_IP")
solr_service_port = os.environ.get("SOLR_SERVICE_PORT")

chatglm_url = os.environ.get("CHATGLM_IP")
chatglm_port = os.environ.get("CHATGLM_PORT")



exp_input = {
    "时间": "01日",
    "全天4G数据流量累计": "6515TB",
    "4G数据流量较日常变化": "-14.8%",
    "4G数据流量较去年变化": "+1.8倍",
    "4G数据流量增幅TOP3": {
        "枣庄": {
            "数据流量": "242TB",
            "较日常": "+26.6%",
            "较去年": "+1.8倍"
        },
        "德州": {
            "数据流量": "345TB",
            "较日常": "+18.4%",
            "较去年": "+2.2倍"
        },
        "菏泽": {
            "数据流量": "528TB",
            "较日常": "+18.3%",
            "较去年": "+1.6倍"
        }
    },
    "4G数据流量降幅TOP3": {
        "青岛": {
            "数据流量": "664TB",
            "较日常": "-45.7%",
            "较去年": "+1.5倍"
        },
        "济南": {
            "数据流量": "528TB",
            "较日常": "-40.1%",
            "较去年": "+1.6倍"
        },
        "烟台": {
            "数据流量": "445TB",
            "较日常": "-34.4%",
            "较去年": "+1.8倍"
        }
    },
    "4G无线接通率": "99.9%",
    "4G无线掉线率": "0.02%",
    "语音业务量": "893万Erl",
    "语音业务量较日常": "+1.8%",
    "语音业务量较去年": "-12.8%",
    "VoLTE语音业务量": "328万Erl",
    "VoLTE语音业务量较日常": "+2.6%",
    "VoLTE语音业务量较去年": "+80.9%",
    "2G无线接通率": "99.8%",
    "VoLTE无线接通率": "99.9%",
    "流量区域较日常变化": {
        "市区": "-34.4%",
        "县城": "-14.3%",
        "乡镇": "-12.4%",
        "农村": "+45.4%"
    },
    "高负荷小区数量": "109个",
    "高负荷小区区域分布": {
        "农村": "43个",
        "市区": "23个",
        "县城": "25个",
        "乡镇": "18个"
    },
    "高负荷小区TOP3地市": {
        "潍坊": "22个",
        "泰安": "13个",
        "济南": "12个"
    }
}

example_input = {
    "_5G消息用户数": {
        "data": {
            "济南": "220万户",
            "青岛": "100户",
            "临沂": "17户"
        },
        "startTime": "2023-06-01 00:00:00"
    },
    "vEPC用户数": {
        "data": {
            "济南": "117万户",
            "青岛": "100户",
            "临沂": "17户"
        },
        "startTime": "2023-06-02 17:30:00"
    },
    "智能网用户数": {
        "data": {
            "济南": "88万户",
            "青岛": "100户",
            "临沂": "17户"
        },
        "startTime": "2023-06-01 00:00:00"
    },
    "_5GToB用户数": {
        "data": {
            "济南": "55户",
            "青岛": "100户",
            "临沂": "17户"
        },
        "startTime": "2023-06-02 17:30:00"
    }
}


class BaseResponse(BaseModel):
    code: int = pydantic.Field(200, description="HTTP status code")
    msg: str = pydantic.Field("success", description="HTTP status message")

    class Config:
        schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
            }
        }


class ListDocsResponse(BaseResponse):
    data: List[str] = pydantic.Field(..., description="List of document names")
    solr: List[str]

    class Config:
        schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
                "data": ["Knowledge_id1", "Knowledge_id2", "Knowledge_id3"],
            }
        }


class ChatMessage(BaseModel):
    question: str = pydantic.Field(..., description="Question text")
    response: str = pydantic.Field(..., description="Response text")
    history: List[List[str]] = pydantic.Field(..., description="History text")
    source_documents: List[str] = pydantic.Field(
        ..., description="List of source documents and their scores"
    )

    # solr
    # solr: List[dict]

    class Config:
        schema_extra = {
            "example": {
                "question": "工伤保险如何办理？",
                "response": "根据已知信息，可以总结如下：\n\n1. 参保单位为员工缴纳工伤保险费，以保障员工在发生工伤时能够获得相应的待遇。\n2. 不同地区的工伤保险缴费规定可能有所不同，需要向当地社保部门咨询以了解具体的缴费标准和规定。\n3. 工伤从业人员及其近亲属需要申请工伤认定，确认享受的待遇资格，并按时缴纳工伤保险费。\n4. 工伤保险待遇包括工伤医疗、康复、辅助器具配置费用、伤残待遇、工亡待遇、一次性工亡补助金等。\n5. 工伤保险待遇领取资格认证包括长期待遇领取人员认证和一次性待遇领取人员认证。\n6. 工伤保险基金支付的待遇项目包括工伤医疗待遇、康复待遇、辅助器具配置费用、一次性工亡补助金、丧葬补助金等。",
                "history": [
                    [
                        "工伤保险是什么？",
                        "工伤保险是指用人单位按照国家规定，为本单位的职工和用人单位的其他人员，缴纳工伤保险费，由保险机构按照国家规定的标准，给予工伤保险待遇的社会保险制度。",
                    ]
                ],
                "source_documents": [
                    "出处 [1] 广州市单位从业的特定人员参加工伤保险办事指引.docx：\n\n\t( 一)  从业单位  (组织)  按“自愿参保”原则，  为未建 立劳动关系的特定从业人员单项参加工伤保险 、缴纳工伤保 险费。",
                    "出处 [2] ...",
                    "出处 [3] ...",
                ],
            }
        }


class ChatMessage2(BaseModel):
    question: str = pydantic.Field(..., description="Question text")
    response: str = pydantic.Field(..., description="Response text")
    history: List[List[str]] = pydantic.Field(..., description="History text")
    source_documents: List[str] = pydantic.Field(
        ..., description="List of source documents and their scores"
    )

    # solr
    solr: List[dict]

    class Config:
        schema_extra = {
            "example": {
                "question": "工伤保险如何办理？",
                "response": "根据已知信息，可以总结如下：\n\n1. 参保单位为员工缴纳工伤保险费，以保障员工在发生工伤时能够获得相应的待遇。\n2. 不同地区的工伤保险缴费规定可能有所不同，需要向当地社保部门咨询以了解具体的缴费标准和规定。\n3. 工伤从业人员及其近亲属需要申请工伤认定，确认享受的待遇资格，并按时缴纳工伤保险费。\n4. 工伤保险待遇包括工伤医疗、康复、辅助器具配置费用、伤残待遇、工亡待遇、一次性工亡补助金等。\n5. 工伤保险待遇领取资格认证包括长期待遇领取人员认证和一次性待遇领取人员认证。\n6. 工伤保险基金支付的待遇项目包括工伤医疗待遇、康复待遇、辅助器具配置费用、一次性工亡补助金、丧葬补助金等。",
                "history": [
                    [
                        "工伤保险是什么？",
                        "工伤保险是指用人单位按照国家规定，为本单位的职工和用人单位的其他人员，缴纳工伤保险费，由保险机构按照国家规定的标准，给予工伤保险待遇的社会保险制度。",
                    ]
                ],
                "source_documents": [
                    "出处 [1] 广州市单位从业的特定人员参加工伤保险办事指引.docx：\n\n\t( 一)  从业单位  (组织)  按“自愿参保”原则，  为未建 立劳动关系的特定从业人员单项参加工伤保险 、缴纳工伤保 险费。",
                    "出处 [2] ...",
                    "出处 [3] ...",
                ],
            }
        }


class SolrChatMessage(BaseModel):
    question: str = pydantic.Field(..., description="Question text")
    response: str = pydantic.Field(..., description="Response text")
    history: List[List[str]] = pydantic.Field(..., description="History text")
    source_documents: List[List[str]] = pydantic.Field(..., description="List of source documents and their scores")
    solr: List[dict]


def get_folder_path(local_doc_id: str):
    return os.path.join(UPLOAD_ROOT_PATH, local_doc_id)


def get_vs_path(local_doc_id: str):
    return os.path.join(VS_ROOT_PATH, local_doc_id)


def get_file_path(local_doc_id: str, doc_name: str):
    return os.path.join(UPLOAD_ROOT_PATH, local_doc_id, doc_name)


# 判断solr中文件是否存在
def is_exist_groupfile(group, filename):
    url = f"http://{solr_fastapi_url}:{solr_fastapi_port}/solr/is_exist_groupfile?group={group}&filename={filename}"
    test_post_response = requests.post(url)
    res = json.loads(test_post_response.content.decode('utf-8'))
    li = [i[0] for i in res["filelist"]]
    if len(li) == 0:
        return False
    return True


''' 1、接口文档 '''


# @app.get("/", response_model=BaseResponse)
async def document():
    return RedirectResponse(url="/docs")


@app.get("/Inspur/docs", tags=["接口文档"])
async def Inspur_Docs():
    res = await document()
    return res


''' 2、闲聊 '''


async def chat(
        question: str = Body(..., description="Question", example="工伤保险是什么？"),
        history: List[List[str]] = Body(
            [],
            description="History of previous questions and answers",
            example=[
                [
                    "工伤保险是什么？",
                    "工伤保险是指用人单位按照国家规定，为本单位的职工和用人单位的其他人员，缴纳工伤保险费，由保险机构按照国家规定的标准，给予工伤保险待遇的社会保险制度。",
                ]
            ],
        ),
):
    for answer_result in local_doc_qa.llm.generatorAnswer(prompt=question, history=history,
                                                          streaming=True):
        resp = answer_result.llm_output["answer"]
        history = answer_result.history
        pass

    #return ChatMessage(
    return SolrChatMessage(
        question=question,
        response=resp,
        history=history,
        source_documents=[],
        solr=[]
    )


#@app.post("/Inspur/chat", response_model=ChatMessage, tags=["闲聊"])
@app.post("/Inspur/chat", response_model=SolrChatMessage, tags=["闲聊"])
async def Inspur_Chat(
        question: str = Body(..., description="Question", example="工伤保险是什么？"),
        history: List[List[str]] = Body(
            [],
            description="History of previous questions and answers",
            example=[
                [
                    "工伤保险是什么？",
                    "工伤保险是指用人单位按照国家规定，为本单位的职工和用人单位的其他人员，缴纳工伤保险费，由保险机构按照国家规定的标准，给予工伤保险待遇的社会保险制度。",
                ]
            ],
        ),
):
    res = await chat(question, history)
    return res


''' 3、bing search 对话 '''


async def bing_search_chat(
        question: str = Body(..., description="Question", example="工伤保险是什么？"),
        history: Optional[List[List[str]]] = Body(
            [],
            description="History of previous questions and answers",
            example=[
                [
                    "工伤保险是什么？",
                    "工伤保险是指用人单位按照国家规定，为本单位的职工和用人单位的其他人员，缴纳工伤保险费，由保险机构按照国家规定的标准，给予工伤保险待遇的社会保险制度。",
                ]
            ],
        ),
):
    for resp, history in local_doc_qa.get_search_result_based_answer(
            query=question, chat_history=history, streaming=True
    ):
        pass
    source_documents = [
        f"""出处 [{inum + 1}] [{doc.metadata["source"]}]({doc.metadata["source"]}) \n\n{doc.page_content}\n\n"""
        for inum, doc in enumerate(resp["source_documents"])
    ]

    return ChatMessage(
        question=question,
        response=resp["result"],
        history=history,
        source_documents=source_documents,
    )


# bing
@app.post("/Inspur/bing_search_chat", response_model=ChatMessage, tags=["微软 bing search 对话"])
async def Inspur_bing_search_chat(
        question: str = Body(..., description="Question", example="工伤保险是什么？"),
        history: Optional[List[List[str]]] = Body(
            [],
            description="History of previous questions and answers",
            example=[
                [
                    "工伤保险是什么？",
                    "工伤保险是指用人单位按照国家规定，为本单位的职工和用人单位的其他人员，缴纳工伤保险费，由保险机构按照国家规定的标准，给予工伤保险待遇的社会保险制度。",
                ]
            ],
        ),
):
    res = await bing_search_chat(question, history)
    return res


async def upload_formatter_txt(group: str, file: UploadFile = File(...)):
    # 首先可以先判断文件是否存在，若不存在再上传
    name = file.filename
    name = name.split(".")[0]
    print("name---->:", name)
    if is_exist_groupfile(group, name):
        return {"status": 1, "message": str(file) + " Documents is exist"}

    #SOLR_URL = 'http://192.168.12.84:8983/solr/gettingstarted/'
    SOLR_URL_UPLOAD = f'http://{solr_service_url}:{solr_service_port}/solr/gettingstarted/'

    # 连接到Solr
    solr = pysolr.Solr(SOLR_URL_UPLOAD, always_commit=True, timeout=60)

    # 读取和格式化 TXT 文件
    content = await file.read()
    if not content:
        raise {"status": 400, "message": "Empty file", "errors": "Empty file"}

    # 提取文件名作为标题
    file_name = file.filename
    title = os.path.splitext(file_name)[0]

    # 逐行读取内容并构建文档列表
    documents = []
    errors = []
    lines = content.decode("utf-8").splitlines()
    topic = None
    for line in lines:
        if line.strip() == '':  # 如果遇到空行则重置topic和content
            topic = None
            continue
        parts = line.strip().split(":", maxsplit=1)
        if len(parts) == 2:
            if '主题' in parts[0].strip():
                topic = parts[1].strip()
            elif '内容' in parts[0].strip() and topic is not None:
                content = parts[1].strip()
                random_uuid = uuid.uuid4()
                # 构建 JSON
                document = {
                    "id": f"{title}_{random_uuid}",
                    "group": group,
                    "group_exact": group,
                    "ttitle": title,
                    "ttitle_exact": title,
                    "description": content,
                    "topic": topic,
                    "topic_exact": topic,
                    "content_type": "text",
                    "create_time": datetime.utcnow().isoformat() + "Z",
                    "update_time": datetime.utcnow().isoformat() + "Z"
                }
                documents.append(document)
                topic = None  # 将topic重置为None以便处理下一个知识体

    def upload_to_solr(document):
        # Do a health check.
        pres = solr.ping()
        # # 定义txt文件的路径
        if pres:
            res = solr.add([document])
            return res
        return False

    # 上传至 Solr
    success_count = 0
    for document in documents:
        try:
            res = upload_to_solr(document)
            # 解析 Solr 响应为 JSON 格式
            response_data = json.loads(res)
            status = response_data["responseHeader"]["status"]
            # 检查添加结果
            if status == 0:
                success_count += 1
            else:
                errors.append(document)
        except Exception as e:
            errors.append(document)

    if success_count == len(documents):
        return {"status": 0, "message": "Documents uploaded successfully"}
    else:
        error_count = len(errors)
        error_message = f"Error uploading {error_count} out of {len(documents)} documents to Solr"
        return {"status": -1, "message": error_message, "errors": errors}


''' 4、单文件上传 '''


async def upload_file(
        file: UploadFile = File(description="A single binary file"),
        knowledge_base_id: str = Form(..., description="Knowledge Base Name", example="kb1"),
):
    # content
    saved_path = get_folder_path(knowledge_base_id)
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    file_content = await file.read()  # 读取上传文件的内容

    file_path = os.path.join(saved_path, file.filename)
    if os.path.exists(file_path) and os.path.getsize(file_path) == len(file_content):
        file_status = f"文件 {file.filename} 已存在。"
        return BaseResponse(code=200, msg=file_status)

    with open(file_path, "wb") as f:
        f.write(file_content)

    # vector
    vs_path = get_vs_path(knowledge_base_id)
    if not os.path.exists(vs_path):
        os.makedirs(vs_path)
    vs_path, loaded_files = local_doc_qa.init_knowledge_vector_store([file_path], vs_path)
    if len(loaded_files) > 0:
        file_status = f"文件 {file.filename} 已上传至新的知识库，并已加载知识库，请开始提问。"
        return BaseResponse(code=200, msg=file_status)
    else:
        file_status = "文件上传失败，请重新上传"
        return BaseResponse(code=500, msg=file_status)


@app.post("/Inspur/upload_file", response_model=BaseResponse, tags=["单文件上传"])
async def Inspur_upload_file(
        file: UploadFile = File(description="A single binary file"),
        knowledge_base_id: str = Form(..., description="Knowledge Base Name", example="kb1"),
):
    _file = copy.deepcopy(file)

    # 先调用/solr/upload_formatter_txt
    test_post_response = await upload_formatter_txt(knowledge_base_id, _file)
    # print(test_post_response)
    if test_post_response["status"] == 0 or test_post_response["status"] == 1:
        res = await upload_file(file, knowledge_base_id)
        return res
    return BaseResponse(code=500, msg="文件上传失败，请重新上传")


''' 5、多文件上传 '''


async def upload_files(
        files: Annotated[
            List[UploadFile], File(description="Multiple files as UploadFile")
        ],
        knowledge_base_id: str = Form(..., description="Knowledge Base Name", example="kb1"),
):
    saved_path = get_folder_path(knowledge_base_id)
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    filelist = []
    for file in files:
        file_content = ''
        file_path = os.path.join(saved_path, file.filename)
        file_content = file.file.read()
        if os.path.exists(file_path) and os.path.getsize(file_path) == len(file_content):
            continue
        with open(file_path, "ab+") as f:
            f.write(file_content)
        filelist.append(file_path)
    if filelist:
        vs_path, loaded_files = local_doc_qa.init_knowledge_vector_store(filelist, get_vs_path(knowledge_base_id))
        if len(loaded_files):
            file_status = f"已上传 {'、'.join([os.path.split(i)[-1] for i in loaded_files])} 至知识库，并已加载知识库，请开始提问"
            return BaseResponse(code=200, msg=file_status)
    file_status = "文件未成功加载，请重新上传文件"
    return BaseResponse(code=500, msg=file_status)


@app.post("/Inspur/upload_files", response_model=BaseResponse, tags=["多文件上传"])
async def Inspur_upload_files(
        files: Annotated[
            List[UploadFile], File(description="Multiple files as UploadFile")
        ],
        knowledge_base_id: str = Form(..., description="Knowledge Base Name", example="kb1"),
):
    # 拷贝一份
    _files = copy.deepcopy(files)

    # 先在solr上传，再chatglm上传
    for file in files:
        test_post_response = await upload_formatter_txt(knowledge_base_id, file)
        if test_post_response["status"] != 0 and test_post_response["status"] != 1:
            return BaseResponse(code=500, msg=str(file) + "在 solr 中文件未成功加载，请检查并修改后重新上传文件！")

    res = await upload_files(_files, knowledge_base_id)
    return res


''' 6、获取已有知识库 '''


@app.get("/Inspur/knowledge_id_list", tags=["获取已有知识库"])
async def Inspur_knowledge_id_list():
    local_knowledge_id_folder = get_folder_path("")

    # 获取知识
    local_knowledge_id_list = os.listdir(local_knowledge_id_folder)

    if len(local_knowledge_id_list) > 0:
        all_knowledge_id_names = [
            doc for doc in local_knowledge_id_list
        ]

        # solr遍历每个knowledge_id
        solr_list = []
        for knowledge_id in all_knowledge_id_names:
            url = f"http://{solr_fastapi_url}:{solr_fastapi_port}/solr/is_exist_group?group={knowledge_id}"
            test_post_response = requests.post(url)
            res = json.loads(test_post_response.content.decode('utf-8'))
            if res["result"] == True:
                solr_list.append(knowledge_id)

        return ListDocsResponse(data=all_knowledge_id_names, solr=solr_list)

    return ListDocsResponse(data=[], solr=[])


''' 7、获取指定知识包含的文件 '''


async def list_docs(
        knowledge_base_id: Optional[str] = Query(default=None, description="Knowledge Base Name", example="kb1")
):
    if knowledge_base_id:
        local_doc_folder = get_folder_path(knowledge_base_id)
        # print(local_doc_folder)
        if not os.path.exists(local_doc_folder):
            return ListDocsResponse(data=[f"Knowledge base {knowledge_base_id} not found"], solr=[])
        all_doc_names = [
            doc
            for doc in os.listdir(local_doc_folder)
            if os.path.isfile(os.path.join(local_doc_folder, doc))
        ]
        return ListDocsResponse(data=all_doc_names, solr=[])
    else:
        if not os.path.exists(UPLOAD_ROOT_PATH):
            all_doc_ids = []
        else:
            all_doc_ids = [
                folder
                for folder in os.listdir(UPLOAD_ROOT_PATH)
                if os.path.isdir(os.path.join(UPLOAD_ROOT_PATH, folder))
            ]

        return ListDocsResponse(data=all_doc_ids, solr=[])


@app.get("/Inspur/list_files", response_model=ListDocsResponse, tags=["获取指定知识下面的文件"])
async def Inspur_list_docs(
        knowledge_base_id: Optional[str] = Query(default=None, description="Knowledge Base Name", example="kb1")
):
    # 调用/solr/is_exist_group
    url = f"http://{solr_fastapi_url}:{solr_fastapi_port}/solr/is_exist_group?group={knowledge_base_id}"
    test_post_response = requests.post(url)
    res = json.loads(test_post_response.content.decode('utf-8'))
    solr_file = [i[0] for i in res["file_list"]]
    res = await list_docs(knowledge_base_id)
    res.solr = solr_file
    return res


''' 8、基于本地知识库对话 '''


async def local_doc_chat(
        knowledge_base_id: str = Body(..., description="Knowledge Base Name", example="kb1"),
        question: str = Body(..., description="Question", example="工伤保险是什么？"),
        history: List[List[str]] = Body(
            [],
            description="History of previous questions and answers",
            example=[
                [
                    "工伤保险是什么？",
                    "工伤保险是指用人单位按照国家规定，为本单位的职工和用人单位的其他人员，缴纳工伤保险费，由保险机构按照国家规定的标准，给予工伤保险待遇的社会保险制度。",
                ]
            ],
        ),
):
    vs_path = os.path.join(VS_ROOT_PATH, knowledge_base_id)
    if not os.path.exists(vs_path):
        return SolrChatMessage(
            question=question,
            response=f"Knowledge base {knowledge_base_id} not found",
            history=history,
            source_documents=[[""]],
            solr=[],
        )
    else:
        for resp, history in local_doc_qa.get_knowledge_based_answer(
                query=question, vs_path=vs_path, chat_history=history, streaming=True
        ):
            pass

        if "无法回答" in resp["result"] or "没有提供足够的相关信息" in resp["result"]:
            chat_resp = await chat(question, history)
            #chat_resp = await Inspur_Chat(question, history)
            #chat_resp.solr = []
            print(111111)
            return chat_resp


        source_documents = [
            f"""出处 [{inum + 1}] {os.path.split(doc.metadata['source'])[-1]}：\n\n{doc.page_content}\n\n"""
            f"""相关度：{doc.metadata['score']}\n\n"""
            for inum, doc in enumerate(resp["source_documents"]) if doc.metadata['score'] < VECTOR_SEARCH_SCORE_THRESHOLD
        ]

        return SolrChatMessage(
            question=question,
            response=resp["result"],
            history=history,
            source_documents=[source_documents],
            solr=[],
        )


@app.post("/Inspur/local_doc_chat", response_model=SolrChatMessage, tags=["基于本地知识库对话"])
async def Inspur_local_doc_chat(
        knowledge_base_id: str = Body(..., description="知识库名称", example="知识库（用中文）"),
        question: str = Body(..., description="Question", example="工伤保险是什么？"),
        history: List[List[str]] = Body(
            [],
            description="History of previous questions and answers",
            example=[
                [
                    "工伤保险是什么？",
                    "工伤保险是指用人单位按照国家规定，为本单位的职工和用人单位的其他人员，缴纳工伤保险费，由保险机构按照国家规定的标准，给予工伤保险待遇的社会保险制度。",
                ]
            ],
        ),
):
    # 调用solr
    #get_test_url = f"http://192.168.12.84:9003/solr/search_topic/?group={knowledge_base_id}&q={question}"
    get_test_url = f"http://{solr_fastapi_url}:{solr_fastapi_port}/solr/search_topic/?group={knowledge_base_id}&q={question}"
    test_post_response = requests.post(get_test_url)
    test_post_response = json.loads(test_post_response.content.decode('utf-8'))
    # print("test_post_response:", test_post_response)
    # print("test_post_response['results']:", test_post_response["results"])

    # 调用chatglm
    res = await local_doc_chat(knowledge_base_id, question, history)
    print("res----->:", res)
    res.solr = test_post_response["results"]
    return res


''' 9、基于所有知识库对话 '''
# @app.post("/Inspur/all_local_doc_chat", response_model=ChatMessage2, tags=["基于所有知识库对话"])
# async def all_local_doc_chat(
#         question: str = Body(..., description="Question", example="工伤保险是什么？"),
#         history: List[List[str]] = Body(
#             [],
#             description="History of previous questions and answers",
#             example=[
#                 [
#                     "工伤保险是什么？",
#                     "工伤保险是指用人单位按照国家规定，为本单位的职工和用人单位的其他人员，缴纳工伤保险费，由保险机构按照国家规定的标准，给予工伤保险待遇的社会保险制度。",
#                 ]
#             ],
#         )
# ):
#     print(VS_ROOT_PATH)
#     print(os.listdir(VS_ROOT_PATH))
#     if not os.listdir(VS_ROOT_PATH):
#         return ChatMessage2(
#             question=question,
#             response=f"Knowledge base is None",
#             history=history,
#             source_documents=[[]],
#         )
#     else:
#         source_documents = []
#         _source_documents = []
#         inum = 0
#         # 获取所有文件路径
#         for i in os.listdir(VS_ROOT_PATH):
#             vs_path = os.path.join(VS_ROOT_PATH, i)
#             logging.info("vs_path:", vs_path)
#             for resp, history in local_doc_qa.get_knowledge_based_answer(
#                     query=question, vs_path=vs_path, chat_history=history, streaming=True
#             ):
#                 logging.info(resp)
#                 pass
#             for doc in resp["source_documents"]:
#                 _source_documents.append(
#                     f"""出处 [{inum + 1}] {os.path.split(doc.metadata['source'])[-1]}：{doc.page_content}"""
#                     f"""相关度：{doc.metadata['score']}"""
#                 )
#                 inum += 1
#             # print("---->", resp["result"])
#
#         # print(len(history), history)
#         source_documents.append(_source_documents)
#
#         return ChatMessage2(
#             question=question,
#             response=resp["result"],
#             history=history,
#             source_documents=source_documents,
#         )


''' 10、删除文件 '''


async def delete_docs(
        knowledge_base_id: str = Query(...,
                                       description="Knowledge Base Name",
                                       example="kb1"),
        doc_name: Optional[str] = Query(
            None, description="doc name", example="doc_name_1.pdf"
        ),
):
    knowledge_base_id = urllib.parse.unquote(knowledge_base_id)
    if not os.path.exists(os.path.join(UPLOAD_ROOT_PATH, knowledge_base_id)):
        return {"code": 1, "msg": f"Knowledge base {knowledge_base_id} not found"}
    if doc_name:
        doc_path = get_file_path(knowledge_base_id, doc_name)
        if os.path.exists(doc_path):
            os.remove(doc_path)

            # 删除上传的文件后重新生成知识库（FAISS）内的数据
            remain_docs = await list_docs(knowledge_base_id)
            if len(remain_docs.data) == 0:
                shutil.rmtree(get_folder_path(knowledge_base_id), ignore_errors=True)
            else:
                local_doc_qa.init_knowledge_vector_store(
                    get_folder_path(knowledge_base_id), get_vs_path(knowledge_base_id)
                )

            return BaseResponse(code=200, msg=f"document {doc_name} delete success")
        else:
            return BaseResponse(code=1, msg=f"document {doc_name} not found")

    else:
        shutil.rmtree(get_folder_path(knowledge_base_id))
        return BaseResponse(code=200, msg=f"Knowledge Base {knowledge_base_id} delete success")


@app.delete("/Inspur/delete_file", response_model=BaseResponse, tags=["删除文件"])
async def Inspur_delete_docs(
        knowledge_base_id: str = Query(...,
                                       description="Knowledge Base Name",
                                       example="kb1"),
        doc_name: Optional[str] = Query(
            None, description="doc name", example="doc_name_1.pdf"
        ),
):
    res = await delete_docs(knowledge_base_id, doc_name)
    if res.code == 200:
        # 先判断solr中是否存在
        filename = doc_name.split(".")[0]
        url = f"http://{solr_fastapi_url}:{solr_fastapi_port}/solr/is_exist_groupfile?group={knowledge_base_id}&filename={filename}"
        test_post_response = requests.post(url)
        res = json.loads(test_post_response.content.decode('utf-8'))
        if res["result"] == False:
            return BaseResponse(code=1, msg=f"document {doc_name} not found from Solr")

        # 若存在，调用/solr/delete_group_title
        url = f"http://{solr_fastapi_url}:{solr_fastapi_port}/solr/delete_group_title?group={knowledge_base_id}&ttitle={filename}"
        test_post_response = requests.post(url)
        res = json.loads(test_post_response.content.decode('utf-8'))
        # print("delete_group_title:", str(res))

    return res


''' 11、删除知识库 '''


@app.delete("/Inspur/delete_Knowledge", response_model=BaseResponse, tags=["删除知识库"])
async def Inspur_delete_Knowledge(
        knowledge_base_id: str = Query(...,
                                       description="Knowledge Base Name",
                                       example="kb1"),
):
    # 1、获取上传文件的路径
    knowledge_base_id_list = os.listdir(UPLOAD_ROOT_PATH)

    # 2、判断是否存在knowledge_base_id
    if knowledge_base_id not in knowledge_base_id_list:
        return {"code": 1, "msg": f"Knowledge base {knowledge_base_id} not found"}

    # 调用/solr/delete_group
    # 3、先判断是否存在（group/knowledge_base_id）
    url = f"http://{solr_fastapi_url}:{solr_fastapi_port}/solr/is_exist_group?group={knowledge_base_id}"
    test_post_response = requests.post(url)
    res = json.loads(test_post_response.content.decode('utf-8'))
    if res["result"] == False:
        return {"code": 1, "msg": f"Knowledge base {knowledge_base_id} not found from Solr"}

    # 4、删除content下面对应的knowledge_base_id文件夹（保存的上传文件）
    knowledge_id_path = os.path.join(UPLOAD_ROOT_PATH, knowledge_base_id)
    # print("knowledge_id_path:", knowledge_id_path)
    shutil.rmtree(knowledge_id_path)

    # 5、删除vector_store下面对应的knowledge_base_id文件夹（保存的文件转换后的向量）
    vs_path = os.path.join(VS_ROOT_PATH, knowledge_base_id)
    # print("vs_path:", vs_path)
    shutil.rmtree(vs_path)

    # 6、删除solr中的group
    solr_del_url = f"http://{solr_fastapi_url}:{solr_fastapi_port}/solr/delete_group?group={knowledge_base_id}"
    del_post_response = requests.post(solr_del_url)
    del_res = json.loads(del_post_response.content.decode('utf-8'))
    logging.info(str(knowledge_base_id) + " " + del_res["result"] + " from solr")

    return BaseResponse(code=200, msg=f"{knowledge_base_id} delete success")


# @app.websocket("/local_doc_qa/stream-chat/{knowledge_base_id}")
async def stream_chat(websocket: WebSocket, knowledge_base_id: str):
    await websocket.accept()
    turn = 1
    while True:
        input_json = await websocket.receive_json()
        question, history, knowledge_base_id = input_json["question"], input_json["history"], input_json[
            "knowledge_base_id"]
        vs_path = os.path.join(VS_ROOT_PATH, knowledge_base_id)

        if not os.path.exists(vs_path):
            await websocket.send_json({"error": f"Knowledge base {knowledge_base_id} not found"})
            await websocket.close()
            return

        await websocket.send_json({"question": question, "turn": turn, "flag": "start"})

        last_print_len = 0
        for resp, history in local_doc_qa.get_knowledge_based_answer(
                query=question, vs_path=vs_path, chat_history=history, streaming=True
        ):
            await websocket.send_text(resp["result"][last_print_len:])
            last_print_len = len(resp["result"])

        source_documents = [
            f"""出处 [{inum + 1}] {os.path.split(doc.metadata['source'])[-1]}：\n\n{doc.page_content}\n\n"""
            f"""相关度：{doc.metadata['score']}\n\n"""
            for inum, doc in enumerate(resp["source_documents"])
        ]

        await websocket.send_text(
            json.dumps(
                {
                    "question": question,
                    "turn": turn,
                    "flag": "end",
                    "sources_documents": source_documents,
                },
                ensure_ascii=False,
            )
        )
        turn += 1


@app.websocket("/Inspur/stream-chat/{knowledge_base_id}")
async def Inspur_stream_chat(websocket: WebSocket, knowledge_base_id: str):
    res = await stream_chat(websocket, knowledge_base_id)
    return res


''' 12、简单简报 '''


@app.post("/Inspur/summary_doc_simple", tags=["简报复杂格式"])
async def summary_doc_s(json_data: Dict[str, Any] = Body(..., description="Json Data", example=example_input)):
    res = await summary_doc_comp(json_data)
    return {"response": res.response}


async def summary_doc_simple(data: str):
    # 获取请求中的问题和历史数据
    json_str = str(data)
    # 去除引号
    json_str = json_str.replace('"', '')
    # 将花括号替换为方括号
    json_str = json_str.replace('{', '[')
    json_str = json_str.replace('}', ']')
    formater = f"任务:性能域简报生成.作为一个维保专家，你可以基于输入的性能数据,生成逻辑清晰、语法正确的简报。提供的信息包括下面类别1. 地市名称、网元名称、资源名称、2. 时间点、时间段、3. 指标名称、指标算法、指标含义、4. 当前值、累计值、预计值、波峰值、波谷值、5. 同比、环比、较昨日、6. 较去年、较去年农历等。请首先针对```符号括起来的文本识别上述性能数据类别,逐个整理分析指标数据（没有出现在上面类别的指标按照给出的类别进行分类）的意义和价值后,输出一段不超过500字的自然文本,不要给出未提供的数字和不实际,并且在结尾增加两个部分:1.监测与优化,扩展内存容量,定期维护相关建议2.总结,总结前面输出的报告内容3.免责声明,免责声明文字不得变动必须与方括号内保持一致 [请注意:本简报数据仅为统计结果,可能会有轻微误差]```{json_str}```".replace(
        "'", "")

    res = await chat(formater, [])
    return res


''' 13、复杂简报 '''


async def summary_doc_comp(data: str):
    # 获取请求中的问题和历史数据
    json_str = str(data)
    # 去除引号
    json_str = json_str.replace('"', '')
    # 将花括号替换为方括号
    json_str = json_str.replace('{', '[')
    json_str = json_str.replace('}', ']')
    doc = f'''
    任务：生成一份网络性能数据报告。
基于方括号[]部分输入原始文本反引号```的json数据，列出所有的指标及其对应的值。
在这份报告中，你无需深度分析数据，只需以自然的文字方式准确地展示出来。
注意，报告应按照json的根节点进行组织，并且字数不应超过1000字。
请不要添加任何不在输入数据中的信息，包括人名、地名、指标信息和数字。
文本输入:
在[[时间]]全天，4G数据流量累计达到[[全天4G数据流量累计]]，较日常有[[4G数据流量较日常变化]]的变化，与去年同期相比增长了[[4G数据流量较去年变化]]。在此期间，数据流量增幅最大的三个地市分别是：枣庄，数据流量为[[4G数据流量增幅TOP3.枣庄.数据流量]]，较日常[[4G数据流量增幅TOP3.枣庄.较日常]]，较去年增长[[4G数据流量增幅TOP3.枣庄.较去年]]；德州，数据流量为[[4G数据流量增幅TOP3.德州.数据流量]]，较日常[[4G数据流量增幅TOP3.德州.较日常]]，较去年增长[[4G数据流量增幅TOP3.德州.较去年]]；菏泽，数据流量为[[4G数据流量增幅TOP3.菏泽.数据流量]]，较日常[[4G数据流量增幅TOP3.菏泽.较日常]]，较去年增长[[4G数据流量增幅TOP3.菏泽.较去年]]。
而数据流量降幅最大的三个地市分别是：青岛，数据流量为[[4G数据流量降幅TOP3.青岛.数据流量]]，较日常[[4G数据流量降幅TOP3.青岛.较日常]]，较去年增长[[4G数据流量降幅TOP3.青岛.较去年]]；济南，数据流量为[[4G数据流量降幅TOP3.济南.数据流量]]，较日常[[4G数据流量降幅TOP3.济南.较日常]]，较去年增长[[4G数据流量降幅TOP3.济南.较去年]]；烟台，数据流量为[[4G数据流量降幅TOP3.烟台.数据流量]]，较日常[[4G数据流量降幅TOP3.烟台.较日常]]，较去年增长[[4G数据流量降幅TOP3.
原始文本:```{json_str}```
    '''.replace("'", "")

    res = await chat(doc, [])
    return res


@app.post("/Inspur/summary_doc_comp", tags=["简报复杂格式"])
async def summary_doc_c(json_data: Dict[str, Any] = Body(..., description="Json Data", example=exp_input)):
    res = await summary_doc_comp(json_data)
    return {"response": res.response}


def api_start(host, port):
    # global app
    global local_doc_qa

    llm_model_ins = shared.loaderLLM()
    llm_model_ins.set_history_len(LLM_HISTORY_LEN)

    # app = FastAPI()
    # Add CORS middleware to allow all origins
    # 在config.py中设置OPEN_DOMAIN=True，允许跨域
    # set OPEN_DOMAIN=True in config.py to allow cross-domain
    if OPEN_CROSS_DOMAIN:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    # app.websocket("/local_doc_qa/stream-chat/{knowledge_base_id}")(stream_chat)
    # app.get("/", response_model=BaseResponse)(document)
    # app.post("/chat", response_model=ChatMessage)(chat)
    # app.post("/local_doc_qa/upload_file", response_model=BaseResponse)(upload_file)
    # app.post("/local_doc_qa/upload_files", response_model=BaseResponse)(upload_files)
    # app.post("/local_doc_qa/local_doc_chat", response_model=ChatMessage)(local_doc_chat)
    # app.post("/local_doc_qa/bing_search_chat", response_model=ChatMessage)(bing_search_chat)
    # app.get("/local_doc_qa/list_files", response_model=ListDocsResponse)(list_docs)
    # app.delete("/local_doc_qa/delete_file", response_model=BaseResponse)(delete_docs)

    local_doc_qa = LocalDocQA()
    local_doc_qa.init_cfg(
        llm_model=llm_model_ins,
        embedding_model=EMBEDDING_MODEL,
        embedding_device=EMBEDDING_DEVICE,
        top_k=VECTOR_SEARCH_TOP_K,
    )
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    #parser.add_argument("--host", type=str, default="0.0.0.0")
    #parser.add_argument("--port", type=int, default=18015)
    parser.add_argument("--host", type=str, default=chatglm_url)
    parser.add_argument("--port", type=int, default=chatglm_port)

    # 初始化消息
    args = None
    args = parser.parse_args()
    args_dict = vars(args)
    shared.loaderCheckPoint = LoaderCheckPoint(args_dict)
    api_start(args.host, args.port)
