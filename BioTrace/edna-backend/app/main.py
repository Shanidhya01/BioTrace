from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import router

app = FastAPI(title="eDNA Biodiversity API")

# Enable frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add router
app.include_router(router, prefix="/api")


@app.get("/")
def root():
    """Root endpoint for health/status checks."""
    return {"message": "eDNA Biodiversity API is running"}
