"""
RAG 엔진 핵심 로직
- 벡터DB에서 유사 문서 검색
- Claude API로 답변 생성
"""

import anthropic
from langchain_community.vectorstores import Chroma

import config
from ingest import get_embedding_function


class RAGEngine:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self.vectorstore = self._load_vectorstore()

    def _load_vectorstore(self):
        """저장된 벡터DB 로드"""
        embedding_fn = get_embedding_function()
        return Chroma(
            persist_directory=config.CHROMA_DB_DIR,
            embedding_function=embedding_fn,
            collection_name="aml_laws",
        )

    def search(self, query: str, top_k: int = None) -> list:
        """
        질문과 유사한 문서 청크 검색

        Returns:
            list of (Document, score) tuples
        """
        # top_k가 지정되지 않으면 전체 문서 검색
        if top_k is None and config.TOP_K == 0:
            k = max(self.vectorstore._collection.count(), 1)
        else:
            k = top_k or max(config.TOP_K, 1)

        results = self.vectorstore.similarity_search_with_relevance_scores(
            query, k=k
        )

        # 임계값 이상만 필터링
        filtered = [
            (doc, score)
            for doc, score in results
            if score >= config.SCORE_THRESHOLD
        ]

        return filtered

    def format_context(self, search_results: list) -> str:
        """검색 결과를 Claude에 전달할 컨텍스트 문자열로 변환"""
        if not search_results:
            return "관련 문서를 찾지 못했습니다."

        context_parts = []
        for i, (doc, score) in enumerate(search_results, 1):
            source = doc.metadata.get("source_file", "알 수 없음")
            page = doc.metadata.get("page", "?")
            context_parts.append(
                f"[출처 {i}] {source} (p.{page}) | 유사도: {score:.2f}\n"
                f"{doc.page_content}\n"
            )

        return "\n---\n".join(context_parts)

    def ask(self, question: str, top_k: int = None) -> dict:
        """
        질문에 대해 RAG 파이프라인 실행

        Returns:
            {
                "answer": str,         # Claude의 답변
                "sources": list,       # 검색된 출처 정보
                "context": str,        # 사용된 컨텍스트
            }
        """
        # 1. 유사 문서 검색
        search_results = self.search(question, top_k)

        # 2. 컨텍스트 구성
        context = self.format_context(search_results)

        # 3. Claude API 호출
        user_message = f"""다음은 AML 관련 법령에서 검색된 내용입니다:

<context>
{context}
</context>

위 법령 내용을 기반으로 다음 질문에 답변해주세요:

<question>
{question}
</question>"""

        response = self.client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=config.MAX_TOKENS,
            system=config.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        answer = response.content[0].text

        # 4. 출처 정보 정리
        sources = []
        for doc, score in search_results:
            sources.append({
                "file": doc.metadata.get("source_file", "알 수 없음"),
                "page": doc.metadata.get("page", "?"),
                "score": round(score, 3),
                "preview": doc.page_content[:150] + "...",
            })

        return {
            "answer": answer,
            "sources": sources,
            "context": context,
        }

    def ask_stream(self, question: str, top_k: int = None):
        """
        스트리밍 방식으로 답변 생성 (Streamlit용)

        Yields:
            str chunks of the answer
        """
        # 1. 유사 문서 검색
        search_results = self.search(question, top_k)

        # 2. 컨텍스트 구성
        context = self.format_context(search_results)

        # 3. Claude API 스트리밍 호출
        user_message = f"""다음은 AML 관련 법령에서 검색된 내용입니다:

<context>
{context}
</context>

위 법령 내용을 기반으로 다음 질문에 답변해주세요:

<question>
{question}
</question>"""

        with self.client.messages.stream(
            model=config.CLAUDE_MODEL,
            max_tokens=config.MAX_TOKENS,
            system=config.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            for text in stream.text_stream:
                yield text

        # 검색 결과도 반환할 수 있도록 속성에 저장
        self._last_search_results = search_results

    @property
    def last_sources(self) -> list:
        """마지막 스트리밍 검색의 출처 정보"""
        results = getattr(self, "_last_search_results", [])
        return [
            {
                "file": doc.metadata.get("source_file", "알 수 없음"),
                "page": doc.metadata.get("page", "?"),
                "score": round(score, 3),
                "preview": doc.page_content[:150] + "...",
            }
            for doc, score in results
        ]
