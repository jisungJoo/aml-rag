"""
AML 법령 Q&A 웹 애플리케이션 (Streamlit)
동료와 공유하여 사용할 수 있는 웹 인터페이스
"""

import os
import tempfile
import streamlit as st
import config

# ── 자동 인제스트 (벡터DB 없으면 자동 생성) ───────────────
if not os.path.exists(config.CHROMA_DB_DIR):
    st.info("📚 첫 실행: 법령 문서 인제스트 중... (2-5분 소요)")
    from ingest import main as run_ingest
    run_ingest()

# ── 페이지 설정 ─────────────────────────────────────────
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 스타일 ──────────────────────────────────────────────
st.markdown("""
<style>
    .source-card {
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
    }
    .stButton > button {
        width: 100%;
        text-align: left;
        padding: 8px 16px;
    }
    .upload-success {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ── RAG 엔진 초기화 ────────────────────────────────────
@st.cache_resource
def init_engine():
    """RAG 엔진을 캐시하여 재사용"""
    try:
        from rag_engine import RAGEngine
        return RAGEngine()
    except Exception as e:
        return None, str(e)


def process_uploaded_pdf(uploaded_file):
    """업로드된 PDF를 처리하여 Document 리스트 반환"""
    import pdfplumber
    from langchain_core.documents import Document

    documents = []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        with pdfplumber.open(tmp_path) as pdf:
            total_pages = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source_file": uploaded_file.name,
                                "page": i + 1,
                                "total_pages": total_pages,
                                "uploaded": True,
                            },
                        )
                    )
    except Exception as e:
        # 폴백: pypdf
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_file"] = uploaded_file.name
                doc.metadata["uploaded"] = True
            documents = docs
        except Exception as e2:
            st.error(f"PDF 읽기 실패: {e2}")
    finally:
        os.unlink(tmp_path)

    return documents


def process_uploaded_xlsx(uploaded_file):
    """업로드된 XLSX를 처리하여 Document 리스트 반환"""
    import pandas as pd
    from langchain_core.documents import Document

    documents = []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        xls = pd.ExcelFile(tmp_path)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(tmp_path, sheet_name=sheet_name)
            # 빈 행/열 제거
            df = df.dropna(how="all").dropna(axis=1, how="all")

            if df.empty:
                continue

            # 시트 전체를 텍스트로 변환
            text = f"[시트: {sheet_name}]\n"
            text += df.to_string(index=False)

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source_file": uploaded_file.name,
                        "sheet": sheet_name,
                        "rows": len(df),
                        "uploaded": True,
                    },
                )
            )

            # 시트가 크면 행 단위로도 분할
            if len(df) > 50:
                chunk_size = 30
                for start in range(0, len(df), chunk_size):
                    chunk_df = df.iloc[start:start + chunk_size]
                    chunk_text = f"[시트: {sheet_name}, 행 {start+1}-{start+len(chunk_df)}]\n"
                    chunk_text += chunk_df.to_string(index=False)
                    documents.append(
                        Document(
                            page_content=chunk_text,
                            metadata={
                                "source_file": uploaded_file.name,
                                "sheet": sheet_name,
                                "rows_range": f"{start+1}-{start+len(chunk_df)}",
                                "uploaded": True,
                            },
                        )
                    )
    except Exception as e:
        st.error(f"XLSX 읽기 실패: {e}")
    finally:
        os.unlink(tmp_path)

    return documents


def add_documents_to_vectordb(documents, engine):
    """문서를 청킹 후 벡터DB에 추가"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=config.SEPARATORS,
        length_function=len,
    )

    chunks = splitter.split_documents(documents)
    if chunks:
        engine.vectorstore.add_documents(chunks)
    return len(chunks)


def check_setup():
    """초기 설정 상태 확인"""
    issues = []
    if not config.ANTHROPIC_API_KEY:
        issues.append("❌ ANTHROPIC_API_KEY가 설정되지 않았습니다. `.env` 파일을 확인하세요.")
    if not os.path.exists(config.CHROMA_DB_DIR):
        issues.append("❌ 벡터DB가 없습니다.")
    return issues


# ── 사이드바 ────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ 설정")

    top_k = st.slider(
        "검색 문서 수",
        min_value=1,
        max_value=20,
        value=max(config.TOP_K, 1),
        help="질문과 유사한 법령 조문을 몇 개까지 검색할지 설정합니다.",
    )

    show_sources = st.checkbox("출처 표시", value=True, help="답변과 함께 참조한 법령 출처를 표시합니다.")
    show_context = st.checkbox("원문 컨텍스트 표시", value=False, help="Claude에 전달된 원문 컨텍스트를 표시합니다.")

    # ── 파일 업로드 섹션 ──────────────────────────────────
    st.markdown("---")
    st.markdown("## 📤 문서 추가 업로드")
    st.caption("PDF 또는 XLSX 파일을 업로드하면 기존 RAG에 추가됩니다.")

    uploaded_files = st.file_uploader(
        "파일 선택",
        type=["pdf", "xlsx"],
        accept_multiple_files=True,
        key="file_uploader",
    )

    if uploaded_files:
        if st.button("📥 벡터DB에 추가", type="primary"):
            engine_result = init_engine()
            if isinstance(engine_result, tuple):
                st.error("RAG 엔진이 초기화되지 않았습니다.")
            else:
                total_chunks = 0
                for uploaded_file in uploaded_files:
                    fname = uploaded_file.name
                    with st.spinner(f"📄 {fname} 처리 중..."):
                        if fname.lower().endswith(".pdf"):
                            docs = process_uploaded_pdf(uploaded_file)
                        elif fname.lower().endswith(".xlsx"):
                            docs = process_uploaded_xlsx(uploaded_file)
                        else:
                            st.warning(f"지원하지 않는 형식: {fname}")
                            continue

                        if docs:
                            n_chunks = add_documents_to_vectordb(docs, engine_result)
                            total_chunks += n_chunks
                            st.success(f"✅ {fname}: {len(docs)}페이지 → {n_chunks}개 청크 추가")
                        else:
                            st.warning(f"⚠️ {fname}: 읽을 수 있는 내용이 없습니다.")

                if total_chunks > 0:
                    st.markdown(
                        f'<div class="upload-success">'
                        f"🎉 총 <strong>{total_chunks}개 청크</strong>가 벡터DB에 추가되었습니다!"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    # 캐시 클리어하여 엔진 재로드
                    init_engine.clear()

    # ── 현재 DB 상태 ──────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📊 벡터DB 현황")
    try:
        engine_check = init_engine()
        if not isinstance(engine_check, tuple):
            count = engine_check.vectorstore._collection.count()
            st.metric("저장된 청크 수", f"{count:,}개")
    except Exception:
        st.caption("DB 정보를 불러올 수 없습니다.")

    # ── 예시 질문 ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📚 예시 질문")

    for q in config.EXAMPLE_QUESTIONS:
        if st.button(q, key=f"example_{q[:20]}"):
            st.session_state["input_question"] = q

    st.markdown("---")
    st.markdown(
        "🔒 이 시스템은 정보 제공 목적이며,\n"
        "법률 자문을 대체하지 않습니다."
    )


# ── 메인 영역 ──────────────────────────────────────────
st.markdown(f"# {config.APP_TITLE}")
st.markdown(config.APP_DESCRIPTION)

# 설정 확인
issues = check_setup()
if issues:
    for issue in issues:
        st.warning(issue)
    if any("❌" in i for i in issues):
        st.stop()

# 엔진 초기화
engine_result = init_engine()
if isinstance(engine_result, tuple):
    st.error(f"RAG 엔진 초기화 실패: {engine_result[1]}")
    st.stop()
engine = engine_result

# ── 대화 기록 관리 ──────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 대화 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and "sources" in msg and show_sources:
            with st.expander("📎 참조 출처", expanded=False):
                for src in msg["sources"]:
                    st.markdown(
                        f'<div class="source-card">'
                        f'<strong>{src["file"]}</strong> (p.{src["page"]}) '
                        f'| 유사도: {src["score"]}<br>'
                        f'<small>{src["preview"]}</small>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

# ── 질문 입력 ──────────────────────────────────────────
default_question = st.session_state.pop("input_question", None)
question = st.chat_input("AML 법령에 대해 질문하세요...", key="chat_input")

if default_question and not question:
    question = default_question

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("🔍 관련 법령 검색 중..."):
            answer_placeholder = st.empty()
            full_answer = ""

            for chunk in engine.ask_stream(question, top_k=top_k):
                full_answer += chunk
                answer_placeholder.markdown(full_answer + "▌")

            answer_placeholder.markdown(full_answer)

        sources = engine.last_sources
        if sources and show_sources:
            with st.expander("📎 참조 출처", expanded=False):
                for src in sources:
                    st.markdown(
                        f'<div class="source-card">'
                        f'<strong>{src["file"]}</strong> (p.{src["page"]}) '
                        f'| 유사도: {src["score"]}<br>'
                        f'<small>{src["preview"]}</small>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        if show_context:
            search_results = engine.search(question, top_k=top_k)
            context = engine.format_context(search_results)
            with st.expander("🔧 원문 컨텍스트 (디버그)", expanded=False):
                st.text(context)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_answer,
        "sources": sources,
    })
