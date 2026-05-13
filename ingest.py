"""
문서 인제스트 스크립트
- data/ 폴더의 PDF/텍스트 파일을 읽어서
- 청킹 → 임베딩 → ChromaDB에 저장
"""

import os
import sys
import glob
from tqdm import tqdm

import config


def get_embedding_function():
    """설정에 따라 임베딩 함수 반환"""
    embed_config = config.EMBEDDING_CONFIGS[config.EMBEDDING_MODEL]

    if embed_config["provider"] == "voyage":
        from langchain_community.embeddings import VoyageEmbeddings

        return VoyageEmbeddings(
            model=embed_config["model_name"],
            voyage_api_key=config.VOYAGE_API_KEY,
        )
    else:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=embed_config["model_name"],
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )


def load_documents():
    """data/ 폴더에서 문서 로드"""
    from langchain_community.document_loaders import PyPDFLoader, TextLoader

    documents = []
    data_dir = config.DATA_DIR

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"📁 '{data_dir}' 폴더를 생성했습니다.")
        print("   이 폴더에 AML 법령 PDF/텍스트 파일을 넣어주세요.")
        sys.exit(1)

    # PDF 파일 로드
    pdf_files = glob.glob(os.path.join(data_dir, "**/*.pdf"), recursive=True)
    for pdf_path in tqdm(pdf_files, desc="📄 PDF 로딩"):
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            # 메타데이터에 파일명 추가
            for doc in docs:
                doc.metadata["source_file"] = os.path.basename(pdf_path)
            documents.extend(docs)
        except Exception as e:
            print(f"  ⚠️ {pdf_path} 로딩 실패: {e}")

    # 텍스트 파일 로드
    txt_files = glob.glob(os.path.join(data_dir, "**/*.txt"), recursive=True)
    for txt_path in tqdm(txt_files, desc="📝 텍스트 로딩"):
        try:
            loader = TextLoader(txt_path, encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_file"] = os.path.basename(txt_path)
            documents.extend(docs)
        except Exception as e:
            print(f"  ⚠️ {txt_path} 로딩 실패: {e}")

    print(f"\n✅ 총 {len(documents)}개 문서 페이지 로드 완료")
    return documents


def split_documents(documents):
    """문서를 청크로 분할"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=config.SEPARATORS,
        length_function=len,
    )

    chunks = splitter.split_documents(documents)
    print(f"✅ 총 {len(chunks)}개 청크 생성")
    return chunks


def create_vector_store(chunks):
    """청크를 벡터DB에 저장"""
    from langchain_community.vectorstores import Chroma

    embedding_fn = get_embedding_function()

    # 기존 DB가 있으면 삭제 후 재생성
    if os.path.exists(config.CHROMA_DB_DIR):
        import shutil
        shutil.rmtree(config.CHROMA_DB_DIR)
        print("🗑️  기존 벡터DB 삭제")

    print("🔄 임베딩 생성 & 벡터DB 저장 중... (첫 실행 시 모델 다운로드에 시간 소요)")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        persist_directory=config.CHROMA_DB_DIR,
        collection_name="aml_laws",
    )

    print(f"✅ 벡터DB 저장 완료: {config.CHROMA_DB_DIR}")
    return vectorstore


def main():
    print("=" * 60)
    print("🏛️  AML 법령 문서 인제스트")
    print("=" * 60)

    # 1. 문서 로드
    documents = load_documents()
    if not documents:
        print("❌ data/ 폴더에 문서가 없습니다.")
        print("   PDF 또는 TXT 파일을 넣어주세요.")
        sys.exit(1)

    # 2. 청킹
    chunks = split_documents(documents)

    # 3. 벡터DB 저장
    create_vector_store(chunks)

    print("\n" + "=" * 60)
    print("🎉 인제스트 완료! 이제 'streamlit run app.py'로 실행하세요.")
    print("=" * 60)


if __name__ == "__main__":
    main()
