"""Implements helper functions to assist evaluation cases where other evaluators are not suitable."""
import json
from datetime import datetime, timezone
from typing import Any, Union
from urllib.parse import urlparse

import requests
from beartype.typing import Dict, List
from playwright.sync_api import CDPSession, Page
import os

DATASET = os.environ["DATASET"]
if DATASET not in ["webarena", "visualwebarena"]:
    raise ValueError("Please set the DATASET environment variable, the possible options are `webarena`, `visualwebarena`")

SHOPPING = os.environ.get("SHOPPING", "")


ACCOUNTS = {
    # "reddit": {"username": "MarvelsGrantMan136", "password": "test1234"},
    "shopping": {
        "username": "emma.lopez@gmail.com",
        "password": "Password.123",
    },
    # "classifieds": {
    #     "username": "blake.sullivan@gmail.com",
    #     "password": "Password.123",
    # },
    # "shopping_site_admin": {"username": "admin", "password": "admin1234"},
    # "shopping_admin": {"username": "admin", "password": "admin1234"},
    # "gitlab": {"username": "byteblaze", "password": "hello1234"},
}

# from llms.providers.openai_utils import (
#     generate_from_openai_chat_completion,
# )

import asyncio
import logging
import os
import random
import time
from typing import Any, Union

import aiolimiter
import openai
from openai import AsyncOpenAI, OpenAI
from tqdm.asyncio import tqdm_asyncio

if "OPENAI_API_BASE" not in os.environ:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    aclient = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
else:
    # Used for running vllm models.
    print("WARNING: Using OPENAI_API_KEY=EMPTY")
    client = OpenAI(
        api_key="EMPTY", base_url=os.environ["OPENAI_API_BASE"]
    )
    aclient = AsyncOpenAI(
        api_key="EMPTY", base_url=os.environ["OPENAI_API_BASE"]
    )


def retry_with_exponential_backoff(  # type: ignore
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 3,
    errors: tuple[Any] = (
        openai.RateLimitError,
        openai.BadRequestError,
        openai.InternalServerError,
    ),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):  # type: ignore
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:

                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


@retry_with_exponential_backoff
def generate_from_openai_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
    num_outputs: int = 1,
) -> Union[str, list[str]]:
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )
    if "OPENAI_API_BASE" in os.environ:
        assert "llama" in model.lower()

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        n=num_outputs
    )
    if num_outputs > 1:
        answer: list[str] = [x.message.content for x in response.choices]
    else:
        answer: str = response.choices[0].message.content

    return answer


class PseudoPage:
    def __init__(self, original_page: Page, url: str):
        self.url = url
        self.original_page = original_page

    def __getattr__(self, attr: str) -> Any:
        # Delegate attribute access to the original page object
        if attr not in ["url"]:
            return getattr(self.original_page, attr)
        else:
            return getattr(self, attr)


def shopping_get_auth_token() -> str:
    response = requests.post(
        url=f"{SHOPPING}/rest/default/V1/integration/admin/token",
        headers={"content-type": "application/json"},
        data=json.dumps(
            {
                "username": ACCOUNTS["shopping_site_admin"]["username"],
                "password": ACCOUNTS["shopping_site_admin"]["password"],
            }
        ),
    )
    token: str = response.json()
    return token


def shopping_get_latest_order_url() -> str:
    """Get the latest order url from the shopping website."""

    header = {
        "Authorization": f"Bearer {shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }

    params = {
        "searchCriteria[sortOrders][0][field]": "created_at",
        "searchCriteria[sortOrders][0][direction]": "DESC",
        "searchCriteria[pageSize]": "1",
    }

    response = requests.get(
        f"{SHOPPING}/rest/V1/orders", params=params, headers=header
    )
    assert response.status_code == 200
    response_obj = response.json()["items"][0]
    order_id = int(response_obj["increment_id"])
    order_url = f"{SHOPPING}/sales/order/view/order_id/{order_id}/"
    return order_url


def shopping_get_sku_latest_review_author(sku: str) -> str:
    """Get the latest review for shopping admin."""
    header = {
        "Authorization": f"Bearer {shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }
    response = requests.get(
        f"{SHOPPING}/rest/V1/products/{sku}/reviews", headers=header
    )
    assert response.status_code == 200
    response_obj = response.json()
    if len(response_obj) == 0:
        return ""
    author: str = response_obj[-1]["nickname"]
    return author


def shopping_get_sku_latest_review_rating(sku: str) -> str:
    """Get the latest review for shopping admin."""
    header = {
        "Authorization": f"Bearer {shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }
    response = requests.get(
        f"{SHOPPING}/rest/V1/products/{sku}/reviews", headers=header
    )
    assert response.status_code == 200
    response_obj = response.json()
    if len(response_obj) == 0:
        return ""
    assert response_obj[0]["ratings"][0]["rating_name"] == "Rating"
    rating: str = str(response_obj[-1]["ratings"][0]["percent"])
    return rating


def shopping_get_sku_latest_review_text(sku: str) -> str:
    """Get the latest review text for shopping admin."""
    header = {
        "Authorization": f"Bearer {shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }
    response = requests.get(
        f"{SHOPPING}/rest/V1/products/{sku}/reviews", headers=header
    )
    assert response.status_code == 200
    response_obj = response.json()
    if len(response_obj) == 0:
        return ""
    text: str = response_obj[-1]["detail"]
    return text


def shopping_get_sku_latest_review_title(sku: str) -> str:
    """Get the latest review title for shopping admin."""
    header = {
        "Authorization": f"Bearer {shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }
    response = requests.get(
        f"{SHOPPING}/rest/V1/products/{sku}/reviews", headers=header
    )
    assert response.status_code == 200
    response_obj = response.json()
    if len(response_obj) == 0:
        return ""
    title: str = response_obj[-1]["title"]
    return title


def shopping_get_sku_product_page_url(sku: str) -> str:
    """Get product page url from sku"""
    header = {
        "Authorization": f"Bearer {shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }
    response = requests.get(
        f"{SHOPPING}/rest/V1/products/{sku}", headers=header
    )
    assert response.status_code == 200
    response_obj = response.json()
    if len(response_obj) == 0:
        return ""
    for custom_attributes in response_obj["custom_attributes"]:
        if custom_attributes["attribute_code"] == "url_key":
            return f"{SHOPPING}/{custom_attributes['value']}.html"
    return ""


def shopping_get_all_product_order(
    page: Page | PseudoPage,
) -> List[Dict[str, str]]:
    """
    Get info of all product in a given order page.

    Example output:
    [
        {
            "name": "Kellogg's Special K Protein Bars, Meal Replacement, Protein Snacks, Value Size, Strawberry, 19oz Box (12 Bars)\nSize\n12 Count (Pack of 1)",
            "options": {
                "Size": "12 Count (Pack of 1)"
            },
            "sku": "B00MXUFL0E",
            "price": "$24.50",
            "qty": "Ordered2",
            "subtotal": "$49.00"
        },
        {
            "name": "Kellogg's Special K Protein Bars, Meal Replacement, Protein Snacks, Value Size, Chocolatey Chip Cookie Dough, 19oz Box (12 Bars)",
            "sku": "B07ZD2PB9F",
            "price": "$42.30",
            "qty": "Ordered2",
            "subtotal": "$84.60"
        }
    ]
    """
    try:
        result = page.evaluate(
            f"""
(() => {{
    try {{
        const products = [...document.querySelector("#my-orders-table").getElementsByTagName('tbody')].map(
            (x) => {{
                return [...x.getElementsByTagName('td')].reduce(function(obj, y) {{
                    const key = y.className.split(' ')[1];
                    obj[key] = y.outerText;
                    // check if options exist
                    if (key === 'name' && y.querySelector('dl')) {{
                        var option_dict = {{}}
                        const options = [...y.querySelector('dl').children];
                        for (let i = 0; i < options.length; i += 2) {{
                            option_dict[options[i].outerText] = options[i+1].outerText;
                        }}
                        obj['options'] = option_dict;
                    }}
                    return obj;
                }}, {{}})
            }}
        );
        return products;
    }} catch (e) {{
        // If any errors are caught, return an empty string
        return e;
        return [];
    }}
}})();
            """
        )
        return result
    except Exception as e:
        result = []

    return result


def shopping_get_order_product_name_list(page: Page | PseudoPage) -> str:
    try:
        products = shopping_get_all_product_order(page)

        return " |OR| ".join([p["name"] for p in products])
    except Exception:
        return ""


def shopping_get_order_product_quantity(
    page: Page | PseudoPage, sku: str
) -> int:
    try:
        if "|OR|" in sku:
            skus = sku.split(" |OR| ")
        else:
            skus = [sku]

        products = shopping_get_all_product_order(page)
        for product in products:
            if product["sku"].strip() in skus:
                # Ordered{qty}
                return int(product["qty"][7:])
        return 0
    except Exception:
        return 0


def shopping_get_order_product_option(
    page: Page | PseudoPage, sku: str, option_name: str
) -> str:
    try:
        products = shopping_get_all_product_order(page)
        for product in products:
            if product["sku"].strip() == sku:
                # Ordered{qty}
                return product["options"][option_name]
        return ""
    except Exception as e:
        return ""


def shopping_get_product_attributes(
    page: Page | PseudoPage, attribute: str
) -> str:
    # Get the values of all cells in the table for the given attribute
    try:
        result = page.evaluate(
            f"""
                (() => {{
                try {{
                    // Create an array of search terms, splitting the string by ' |OR| '
                    const searchTerms = '{attribute}'.toLowerCase().split(' |or| ');
                    // Convert the children of the tbody inside the element with the given ID into an array
                    return Array.from(
                    document.querySelector('#productDetails_detailBullets_sections1 > tbody').children
                    )
                    // Filter the array to only include elements where the first child's text includes any of the search terms
                    .filter(x =>
                    searchTerms.some(term => x.children[0].outerText.toLowerCase().includes(term))
                    )
                    // Map over the filtered elements to get the outerText of their second child
                    .map(x => x.children[1].outerText)
                    // Join all the resulting strings with a comma and a space
                    .join(', ')
                }} catch (e) {{
                    // If any errors are caught, return an empty string
                    return ''
                }}
                }})();
            """
        )
    except Exception:
        result = ""

    return result


def shopping_get_product_price(page: Page | PseudoPage) -> Union[float, int]:
    """Get the price of the product on the shopping website."""
    try:
        result = page.evaluate(
            """
                (() => {{
                    res = parseFloat(document.querySelector(\"#maincontent > div.columns > div > div.product-info-main > div.product-info-price > div.price-box.price-final_price > span > span\")
                    .outerText.substr(1));
                    return res ? res : 0;
                }})();
            """
        )
    except Exception:
        result = 0

    return result


def shopping_get_num_reviews(page: Page | PseudoPage) -> int:
    """Get the price of the product on the shopping website."""
    try:
        result = page.evaluate(
            """
                (() => {{
                    res = parseInt(document.querySelector(\"#tab-label-reviews-title\")
                    .outerText.split(' ')[1]);
                    return res ? res : 0; }}
                )();
            """
        )
    except Exception:
        result = 0

    return result


def shopping_get_rating_as_percentage(page: Page | PseudoPage) -> int:
    """Get the rating of the product on the shopping website as a percentage out of 100."""
    try:
        rating = page.evaluate(
            """
                (() => {{
                    ratingPercentage = parseFloat(document.querySelector('.rating-result').title.replace('%', ''));
                    return ratingPercentage ? ratingPercentage : 0;
                }})();
            """
        )
    except Exception:
        rating = 0

    return rating


def get_query_text(page: Page | PseudoPage, selector: str) -> str:
    """Get the text content of the element matching the given selector.

    Note that this function DOES NOT perform downcasing.
    """
    try:
        result = page.evaluate(
            f"""
                (() => {{
                    try {{
                        return document.querySelector('{selector}').textContent;
                    }} catch (e) {{
                        return '';
                    }}
                }})();
            """
        )
    except Exception:
        result = ""

    return result


def get_query_text_lowercase(page: Page | PseudoPage, selector: str) -> str:
    """Get the lowercase text content of the element matching the given selector."""
    return get_query_text(page, selector).lower()



def llm_fuzzy_match(pred: str, reference: str, question: str) -> float:
    """Check whether the prediction matches the reference with GPT-4-turbo"""
    messages: list[dict[str, Any]] = []
    # construct the question to ask
    message = "Help a teacher to grade the answer of a student given a question. Keep in mind that the student may use different phrasing or wording to answer the question. The goal is to evaluate whether the answer is semantically equivalent to the reference answer.\n"
    message += f"question: {question}\n"
    message += f"reference answer: {reference}\n"
    message += "all the string 'N/A' that you see is a special sequence that means 'not achievable'\n"
    message += f"student answer: {pred}\n"
    message += "Conclude the judgement by 'correct', 'incorrect', or 'partially correct'. Only output one of these options, and nothing else."
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": message},
    ]

    response = generate_from_openai_chat_completion(
        model="gpt-4-1106-preview",
        messages=messages,
        temperature=0,
        max_tokens=768,
        top_p=1.0,
        context_length=0,
    ).lower()
    if "partially correct" in response or "incorrect" in response:
        return 0.0
    else:
        assert "correct" in response, response
        return 1.0


def llm_ua_match(pred: str, reference: str, question: str) -> float:
    """Check whether the prediction matches the reference with GPT-4-turbo"""
    messages: list[dict[str, Any]] = []
    # construct the question to ask
    message = ""
    message += f"task: {question}\n"
    message += f"actual unachievable reason: {reference}\n"
    message += f"reported unachievable reason: {pred}\n"
    message += (
        "The task described above is inherently unachievable due to the reason specified under 'actual unachievable reason'. "
        "An individual previously attempted this task and was unable to complete it. They provided a reason for their failure, "
        "which is listed under 'reported unachievable reason'. Your role is to review both the actual and reported reasons. "
        "Determine if the reported reason aligns with the actual reason, even if implicitly. "
        "If the stated reason is in line with the actual reason, respond with 'same'. Otherwise, respond with 'different'."
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": message},
    ]

    response = generate_from_openai_chat_completion(
        model="gpt-4-1106-preview",
        messages=messages,
        temperature=0,
        max_tokens=768,
        top_p=1.0,
        context_length=0,
    ).lower()
    if "different" in response:
        return 0.0
    else:
        assert "same" in response
        return 1.0
