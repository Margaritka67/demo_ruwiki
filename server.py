import os
import sys
import asyncio
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel

import wiki_summarize

app = FastAPI(title="Wiki Summarizer Service", version="1.0")

class SummaryResponse(BaseModel):
    title: str
    summary: str
    new_article_name: str


@app.get("/summarize", response_model=SummaryResponse)
async def summarize(
    title: str = Query(..., description="Название статьи (например, 'Изотопы')"),
    site: str = Query("ru.ruwiki.ru", description="Домен MediaWiki"),
    path: str = Query("/w/", description="Путь к API (по умолчанию /w/)"),
    username: str | None = Query(None, description="Username"),
    password: str | None = Query(None, description="Password"),
):
    """Вызывает wiki_summarize.run_summarization() и wiki_summarize.publish_draft(); возвращает текст и ссылку на опубликованную статью"""
    try:
        summary_text = await asyncio.to_thread(
            wiki_summarize.run_summarization, 
            title, 
            site, 
            path,
        )
        
        new_article_name = await asyncio.to_thread(
            wiki_summarize.publish_draft, 
            summary_text, 
            f"{title}_draft", 
            site, 
            path, 
            username, 
            password, 
            f"Публикация черновика для «{title}_draft»", 
            True,
        )
        
    except wiki_summarize.PageNotFoundError:
        raise HTTPException(status_code=404, detail="Статья не найдена")
    except wiki_summarize.WikiConnectionError as e:
        raise HTTPException(status_code=502, detail=f"Ошибка подключения к Wiki: {e}")
    except wiki_summarize.OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"Ошибка OpenAI API: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Неизвестная ошибка: {e}")

    return SummaryResponse(title=title, summary=summary_text, new_article_name=new_article_name)
