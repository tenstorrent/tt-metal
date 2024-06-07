from fastapi import FastAPI, Request, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import uvicorn
from typing import List

app = FastAPI()


origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Operation(BaseModel):
    id: int
    name: str
    description: str


@app.get("/api")
async def read_root():
    return {"message": "Hello from FastAPI"}


@app.get("/api/get-operations")
async def get_operations(param: str = Query(None)):
    # temp data
    operations = [
        Operation(id=1, name="Operation1", description="Description for operation 1"),
        Operation(id=2, name="Operation2", description="Description for operation 2"),
        Operation(id=3, name="Operation3", description="Description for operation 3"),
    ]
    return JSONResponse(content=[operation.dict() for operation in operations])

# Middleware to proxy requests to Vite development server, excluding /api/* requests
@app.middleware("http")
async def proxy_middleware(request: Request, call_next):
    if request.url.path.startswith("/api"):
        response = await call_next(request)
        return response

    vite_url = f"http://localhost:5173{request.url.path}"
    async with httpx.AsyncClient() as client:
        try:
            vite_response = await client.request(
                method=request.method,
                url=vite_url,
                content=await request.body(),
                headers=request.headers,
                params=request.query_params,
            )
            headers = {k: v for k, v in vite_response.headers.items() if k.lower() != 'content-encoding'}
            return StreamingResponse(vite_response.aiter_bytes(), headers=headers)
        except httpx.RequestError as exc:
            return JSONResponse(status_code=500, content={"message": f"Error connecting to Vite server: {exc}"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
