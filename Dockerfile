FROM python:3.11-slim

# Hugging Face Spaces runs the container as user 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /app
ENV TORCH_THREADS=2

# CPU-only PyTorch (the default PyPI wheel pulls multiple GB of CUDA libraries)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
EXPOSE 10000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]