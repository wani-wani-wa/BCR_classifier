import pytest
from transformers import AutoTokenizer


@pytest.fixture(scope="session")
def esm_tokenizer():
    return AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
