from pydantic import BaseModel, ConfigDict, Field
#BaseModel 是 Pydantic 的数据模型基类；继承它后，QuestionRequest 就能执行字段校验；ConfigDict定义模型配置
from typing import Any, Literal

#QuestionRequest 描述问题请求：问题文本
class QuestionRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, str_strip_whitespace=True,
        #extra="forbid" 拒绝未声明字段，strict=True 要求类型符合声明，str_strip_whitespace=True 去掉字符串两端空白
    )
    question: str = Field(min_length=1, max_length=2000) #question: str 声明问题必须是字符串；没有默认值，所以必须提供
    #Field 添加约束：去掉两端空白后，长度须为1至2000个字符。2000是本接口的输入上限，不是模型的 token 上限。

#Citation 描述一条引用：证据编号、块身份、文件名、资料身份、哈希、物理页码和证据文本，全部来自现有契约解析结果
class Citation(BaseModel):
    evidence_id: str
    chunk_id: str
    source: str
    source_id: str
    content_hash: str
    page: int = Field(ge=1) #要求页码大于等于1
    text: str

#QuestionResponse 描述回答：提示版本、状态、回答状态、回答文本、引用列表、模型名和使用情况
class QuestionResponse(BaseModel):
    model_config = ConfigDict(extra="ignore") #extra="ignore" 过滤未声明的响应字段
    prompt_version: str
    status: Literal["generated", "no_evidence"]
    answer_status: Literal["answered", "refused"]
    answer: str
    evidence_ids: list[str]
    citations: list[Citation] #要求每条引用符合上面的模型
    model: str | None #表示字符串或空值，这里的 | 是类型联合
    usage: dict[str, Any] | None #允许不同供应商的用量明细保留各自结构