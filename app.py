"""
AML 법령 Q&A 웹 애플리케이션 (Streamlit)
동료와 공유하여 사용할 수 있는 웹 인터페이스
"""

import os
import streamlit as st
import config

# ── 자동 인제스트 (벡터DB 없으면 자동 생성) ───────────
if not os.path.exists(config.CHROMA_DB_DIR):
    with st.spinner("📚 첫 실행: 법령 문서 인제스트 중... (2-5분 소요)"):
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
    /* 메인 헤더 */
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    /* 출처 카드 */
    .source-card {
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
    }

    /* 예시 질문 버튼 */
    .stButton > button {
        width: 100%;
        text-align: left;
        padding: 8px 16px;
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


def check_setup():
    """초기 설정 상태 확인"""
    issues = []

    if not config.ANTHROPIC_API_KEY:
        issues.append("❌ ANTHROPIC_API_KEY가 설정되지 않았습니다. `.env` 파일을 확인하세요.")

    if not os.path.exists(config.CHROMA_DB_DIR):
        issues.append("❌ 벡터DB가 없습니다. 먼저 `python ingest.py`를 실행하세요.")

    if not os.path.exists(config.DATA_DIR) or not os.listdir(config.DATA_DIR):
        issues.append("⚠️ `data/` 폴더에 법령 문서가 없습니다.")

    return issues


# ── 사이드바 ────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ 설정")

    top_k = st.slider(
        "검색 문서 수",
        min_value=1,
        max_value=10,
        value=config.TOP_K,
        help="질문과 유사한 법령 조문을 몇 개까지 검색할지 설정합니다.",
    )

    show_sources = st.checkbox("출처 표시", value=True, help="답변과 함께 참조한 법령 출처를 표시합니다.")
    show_context = st.checkbox("원문 컨텍스트 표시", value=False, help="Claude에 전달된 원문 컨텍스트를 표시합니다.")

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

        # 출처 표시
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
# 예시 질문 선택 시 자동 입력
default_question = st.session_state.pop("input_question", None)

question = st.chat_input("AML 법령에 대해 질문하세요...", key="chat_input")

# 예시 질문이 선택된 경우
if default_question and not question:
    question = default_question

if question:
    # 사용자 메시지 표시 & 저장
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # AI 답변 생성
    with st.chat_message("assistant"):
        with st.spinner("🔍 관련 법령 검색 중..."):
            # 스트리밍 답변
            answer_placeholder = st.empty()
            full_answer = ""

            for chunk in engine.ask_stream(question, top_k=top_k):
                full_answer += chunk
                answer_placeholder.markdown(full_answer + "▌")

            answer_placeholder.markdown(full_answer)

        # 출처 표시
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

        # 컨텍스트 표시 (디버그용)
        if show_context:
            search_results = engine.search(question, top_k=top_k)
            context = engine.format_context(search_results)
            with st.expander("🔧 원문 컨텍스트 (디버그)", expanded=False):
                st.text(context)

    # 대화 기록 저장
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_answer,
        "sources": sources,
    })
