from contextlib import contextmanager
from typing import Any, Dict, List, Optional
from langchain_core.runnables import Runnable

try:
    from langsmith import trace as ls_trace
    _LANGSMITH_AVAILABLE = True
except ImportError:
    _LANGSMITH_AVAILABLE = False

try:
    from langfuse import get_client as lf_get_client
    _LANGFUSE_AVAILABLE = True
except ImportError:
    lf_get_client = None
    _LANGFUSE_AVAILABLE = False


def _init_chain_context(ctx: Optional[dict]) -> dict:
    """Initialize or ensure a chain_context dict has the required sub-dicts."""
    if ctx is None:
        ctx = {}
    ctx.setdefault("duration", {})
    ctx.setdefault("reasoning", {})
    ctx.setdefault("tokens", {})
    ctx.setdefault("memory", None)
    return ctx


class Chain(Runnable):
    """
    Compatibility base class that mimics the legacy langchain.chains.base.Chain
    interface (removed in langchain 1.x).

    Subclasses should implement:
      - ``_call(self, inputs, run_manager=None) -> dict``
      - ``input_keys`` property
      - ``output_keys`` property
    """

    def __init__(self, **kwargs):
        self._received_log_ctx = None
        self._received_chain_context = None

    @property
    def input_keys(self) -> List[str]:
        return []

    @property
    def output_keys(self) -> List[str]:
        return []

    def _call(self, inputs: Dict[str, Any], run_manager=None) -> Dict[str, Any]:
        raise NotImplementedError

    @contextmanager
    def _langfuse_span(self, input: Dict[str, Any]):
        """Wrap this chain's execution in a named Langfuse span, so nested chains
        (e.g. ExecuteTimbrQueryChain, GenerateAnswerChain) show up as distinct,
        named observations in the trace instead of being flattened into the parent."""
        if not _LANGFUSE_AVAILABLE:
            yield None
            return

        try:
            client = lf_get_client()
            with client.start_as_current_observation(
                name=self.__class__.__name__, as_type="chain", input=input,
            ) as span:
                yield span
        except Exception:
            yield None

    def invoke(self, input: Dict[str, Any], config=None, log_ctx=None, **kwargs) -> Dict[str, Any]:
        self._received_log_ctx = log_ctx
        self._received_chain_context = _init_chain_context(input.get("chain_context"))
        with self._langfuse_span(input) as lf_span:
            if _LANGSMITH_AVAILABLE:
                with ls_trace(name=self.__class__.__name__, run_type="chain", inputs={"input": input}) as rt:
                    result = self._call(input)
                    result["chain_context"] = self._received_chain_context
                    rt.end(outputs=result)
            else:
                result = self._call(input)
                result["chain_context"] = self._received_chain_context

            if lf_span is not None:
                try:
                    lf_span.update(output=result)
                except Exception:
                    pass

        return result

    async def ainvoke(self, input: Dict[str, Any], config=None, log_ctx=None, **kwargs) -> Dict[str, Any]:
        self._received_log_ctx = log_ctx
        self._received_chain_context = _init_chain_context(input.get("chain_context"))
        with self._langfuse_span(input) as lf_span:
            if _LANGSMITH_AVAILABLE:
                with ls_trace(name=self.__class__.__name__, run_type="chain", inputs={"input": input}) as rt:
                    result = self._call(input)
                    result["chain_context"] = self._received_chain_context
                    rt.end(outputs=result)
            else:
                result = self._call(input)
                result["chain_context"] = self._received_chain_context

            if lf_span is not None:
                try:
                    lf_span.update(output=result)
                except Exception:
                    pass

        return result
