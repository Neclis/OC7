from pydantic import BaseModel
# 2. Class which describes main features data
class datamodel(BaseModel):
    EXT_SOURCE_1 : float
    EXT_SOURCE_2 : float
    EXT_SOURCE_3 : float
    CODE_GENDER : float
    NAME_FAMILY_STATUS : float
    AMT_REQ_CREDIT_BUREAU_TOTAL : float
    BIRTH_EMPLOYED_INTERVEL : float
    AMT_INCOME_TOTAL : float
    AMT_GOODS_PRICE : float
    AMT_CREDIT_SUM_DEBT : float