import asyncio
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
from typing import (
    AsyncIterable,
    Awaitable,
    Callable,
    Generic,
    Iterable,
    Iterator,
    List,
    Set,
    TypeVar,
    Union,
    cast,
)

import pytest

from streamable import (
    Observation,
    star,
    stream,
    stream_aiterator,
    stream_iterator,
)
from streamable._tools._func import AsyncFunction
from tests.tools.error import TestError
from tests.tools.func import (
    async_throw_func,
    audit_awaitable,
    identity,
    nothing,
    throw_func,
)
from tests.tools.iter import (
    ITERABLE_TYPES,
    IterableType,
    aclose_or_close,
    aidentity_or_identity,
    aiter_or_iter,
    aiterable_or_iterable,
    alist_or_list,
    anext_or_next,
    await_or_call,
    stopiteration_type,
)
from tests.tools.loop import (
    TEST_LOOP,
    assert_only_current_task_is_alive,
    max_event_loop_stall,
)
from tests.tools.monitor import (
    SLOW,
    MonitoredIdentity,
    SlowToCancelMonitoredIntAsyncIterator,
    create_close_failing_monitored_int_iterable,
    create_monitored_int_iterable,
    create_monitored_int_iterator,
)

T = TypeVar("T")


def chain(
    src: Union[Iterable[int], AsyncIterable[int]], itype: IterableType
) -> "stream[int]":
    """Every operator flavor, downstream of a `.buffer` pulling `src` ahead."""

    # a sync iteration involving no coroutine function goes through the sync operators
    def get_group(key: int, group: List[int]) -> List[int]:
        return group

    return (
        stream(src)
        .buffer()
        .catch(TestError)
        .do(nothing)
        .filter(lambda _: True)
        .group(1)
        .flatten()
        .group(1, by=identity)
        .map(star(get_group))
        .flatten(concurrency=2)
        .group(1, within=timedelta(hours=1))
        .map(stream)
        .flatten(concurrency=2)
        .group(1, by=identity, within=timedelta(hours=1))
        .map(star(get_group))
        .flatten()
        .map(aidentity_or_identity(itype), concurrency=2)
        .cast(int)
        .map(aidentity_or_identity(itype), concurrency=2, as_completed=True)
        .cast(int)
        .map(identity, concurrency=2)
        .map(identity, concurrency=2, as_completed=True)
        .observe(do=nothing)
        .observe(every=2, do=nothing)
        .observe(every=timedelta(hours=1), do=nothing)
        .skip(1)
        .skip(until=lambda _: True)
        .take(1_000)
        .take(until=lambda _: False)
        .throttle(1_000, per=timedelta(seconds=1))
    )


#############
# ASSERTION #
#############


def assert_no_new_threads(threads: Set[threading.Thread]) -> None:
    # a thread signaling it is done may take a moment to leave `threading.enumerate()`
    deadline = time.perf_counter() + SLOW / 10
    while set(threading.enumerate()) != threads and time.perf_counter() < deadline:
        time.sleep(SLOW / 1000)
    assert set(threading.enumerate()) == threads


def assert_nothing_in_flight(monitored: MonitoredIdentity) -> None:
    # every started pull/call is either completed or cancelled
    assert sorted(monitored.completed + monitored.cancelled) == sorted(
        monitored.started
    )


################
# TERMINATIONS #
################


async def _aexhaust(it: stream_aiterator) -> None:
    with pytest.raises(StopAsyncIteration):
        while True:
            await it.__anext__()


def _exhaust(it: stream_iterator) -> None:
    with pytest.raises(StopIteration):
        while True:
            next(it)


async def _aclose(it: stream_aiterator) -> None:
    await it.aclose()


def _close(it: stream_iterator) -> None:
    it.close()


async def _anext_until_timeout_then_close(it: stream_aiterator) -> None:
    # a timeout does not end the iteration, the close does
    with pytest.raises(asyncio.TimeoutError):
        while True:
            await asyncio.wait_for(it.__anext__(), timeout=SLOW / 10)
    await it.aclose()


It = TypeVar("It", bound=Union[stream_iterator, stream_aiterator])


@dataclass(frozen=True)
class TerminationCase(ABC, Generic[It]):
    @property
    @abstractmethod
    def itype(self) -> IterableType: ...

    @abstractmethod
    def terminate_without_leaks(self, s: "stream", n_pulls: int) -> It: ...


@dataclass(frozen=True)
class SyncTerminationCase(TerminationCase[stream_iterator]):
    terminate: Callable[[stream_iterator], None]

    @property
    def itype(self) -> IterableType:
        return "sync"

    def terminate_without_leaks(self, s: "stream", n_pulls: int) -> stream_iterator:
        threads = set(threading.enumerate())
        it = s.__iter__()
        for _ in range(n_pulls):
            next(it)
        # lets the work scheduled by the last pull start in its thread
        time.sleep(SLOW / 10)
        self.terminate(it)
        assert_no_new_threads(threads)
        it.close()  # noop
        return it


@dataclass(frozen=True)
class AsyncTerminationCase(TerminationCase[stream_aiterator]):
    terminate: AsyncFunction[stream_aiterator, None]

    @property
    def itype(self) -> IterableType:
        return "async"

    async def aterminate_without_leaks(
        self, s: "stream", n_pulls: int
    ) -> stream_aiterator:
        async def pull_then_terminate() -> stream_aiterator:
            it = s.__aiter__()
            for _ in range(n_pulls):
                await it.__anext__()
            # lets the work scheduled by the last pull start
            await asyncio.sleep(0)
            await self.terminate(it)
            return it

        threads = set(threading.enumerate())
        audit = await audit_awaitable(pull_then_terminate())
        assert audit.leftover_tasks == 0
        assert_no_new_threads(threads)
        await audit.result.aclose()
        return audit.result

    def terminate_without_leaks(self, s: "stream", n_pulls: int) -> stream_aiterator:
        return TEST_LOOP.run_until_complete(self.aterminate_without_leaks(s, n_pulls))


EXHAUST = SyncTerminationCase(_exhaust)
AEXHAUST = AsyncTerminationCase(_aexhaust)

CLOSE = SyncTerminationCase(_close)
ACLOSE = AsyncTerminationCase(_aclose)

NEXT_UNTIL_TIMEOUT_THEN_CLOSE = AsyncTerminationCase(_anext_until_timeout_then_close)

TERMINATIONS: List[TerminationCase] = [
    EXHAUST,
    AEXHAUST,
    CLOSE,
    ACLOSE,
    NEXT_UNTIL_TIMEOUT_THEN_CLOSE,
]

#########
# TESTS #
#########


@pytest.mark.parametrize(
    "s",
    [
        stream(range(4)),
        stream(range(4)) + range(4),
        stream(range(4)).buffer(2),
        stream(range(4)).catch(ValueError),
        stream(range(4)).do(nothing),
        stream(range(4)).do(identity, concurrency=2),
        stream(range(4)).filter(),
        stream([range(2), range(2)]).flatten(),
        stream([range(2), range(2)]).flatten(concurrency=2),
        stream(range(4)).group(2),
        stream(range(4)).group(by=nothing),
        stream(range(4)).group(within=timedelta(seconds=1)),
        stream(range(4)).group(by=nothing, within=timedelta(seconds=1)),
        stream(range(4)).map(identity),
        stream(range(4)).map(identity, concurrency=2),
        stream(range(4)).observe(do=nothing),
        stream(range(4)).observe(every=2, do=nothing),
        stream(range(4)).observe(every=timedelta(seconds=1), do=nothing),
        stream(range(4)).skip(1),
        stream(range(4)).skip(until=identity),
        stream(range(4)).take(2),
        stream(range(4)).take(until=nothing),
        stream(range(4)).throttle(2, per=timedelta(seconds=1)),
        stream(range(4)).do(asyncio.sleep, concurrency=2),
        stream(range(4)).map(asyncio.sleep, concurrency=2),
    ],
)
@pytest.mark.parametrize("close_termination", [CLOSE, ACLOSE])
@pytest.mark.parametrize("n_yields", [0, 1])
def test_close_is_idempotent(
    close_termination: TerminationCase, s: stream, n_yields: int
) -> None:
    # the second close is done by `terminate_without_leaks`
    close_termination.terminate_without_leaks(s, n_yields)


@pytest.mark.parametrize(
    "s",
    [
        stream(range(4)).buffer(2),
        stream(range(4)).do(identity, concurrency=2),
        stream([range(2), range(2)]).flatten(concurrency=2),
        stream(range(4)).group(within=timedelta(seconds=1)),
        stream(range(4)).group(by=nothing, within=timedelta(seconds=1)),
        stream(range(4)).map(identity, concurrency=2),
        stream(range(4)).map(identity, concurrency=2, as_completed=True),
        stream(range(4)).do(asyncio.sleep, concurrency=2),
        stream(range(4)).map(asyncio.sleep, concurrency=2),
        stream(range(4)).map(asyncio.sleep, concurrency=2, as_completed=True),
    ],
)
@pytest.mark.parametrize("close_termination", [CLOSE, ACLOSE])
@pytest.mark.parametrize("n_yields", [0, 1])
def test_close_stops_iteration(
    close_termination: TerminationCase, s: stream, n_yields: int
) -> None:
    it = close_termination.terminate_without_leaks(s, n_yields)
    with pytest.raises(stopiteration_type(close_termination.itype)):
        anext_or_next(it, close_termination.itype)


@pytest.mark.parametrize(
    "get_stream, expected",
    [
        (lambda slow: stream(range(6)).map(slow).buffer(2), [0, 1, 2, 3, 4, 5]),
        (lambda slow: stream(range(6)).map(slow, concurrency=2), [0, 1, 2, 3, 4, 5]),
        (
            lambda slow: stream(range(6)).map(slow).group(2, within=timedelta(hours=1)),
            [[0, 1], [2, 3], [4, 5]],
        ),
        (
            lambda slow: stream(range(6))
            .group(3)
            .map(lambda g: stream(g).map(slow))
            .flatten(concurrency=2),
            [0, 3, 1, 4, 2, 5],
        ),
    ],
)
@pytest.mark.asyncio
async def test_timeouts_do_not_end_the_iteration(
    get_stream: Callable[[Callable[[int], Awaitable[int]]], "stream"],
    expected: List,
) -> None:
    monitored_identity = MonitoredIdentity[int](n_fast=0)
    it = get_stream(monitored_identity.async_func).__aiter__()
    elems: List = []
    n_timeouts = 0
    while True:
        try:
            elems.append(await asyncio.wait_for(it.__anext__(), timeout=SLOW / 3))
        except asyncio.TimeoutError:
            n_timeouts += 1
        except StopAsyncIteration:
            break
    assert n_timeouts
    # retrying after timeouts loses nothing
    assert elems == expected
    # in-flight work is never cancelled
    assert monitored_identity.cancelled == []
    await it.aclose()
    assert_only_current_task_is_alive()


@pytest.mark.parametrize(
    "termination, completed, cancelled",
    [
        (EXHAUST, [0, 1], []),
        (AEXHAUST, [0, 1], []),
        (CLOSE, [0], []),
        (ACLOSE, [0], []),
        (NEXT_UNTIL_TIMEOUT_THEN_CLOSE, [0], [1]),
    ],
)
def test_map_termination(
    termination: TerminationCase, completed: List, cancelled: List
) -> None:
    src = create_monitored_int_iterator(termination.itype, n_yields=2)
    s = stream(src).map(identity)
    termination.terminate_without_leaks(s, n_pulls=1)
    assert src.completed == completed
    assert src.cancelled == cancelled
    # not owned src iterator, not closed
    assert not src.close_attempted


@pytest.mark.parametrize(
    "termination, completed, cancelled",
    [
        (EXHAUST, [0, 1, 2, 3], []),
        (AEXHAUST, [0, 1, 2, 3], []),
        (CLOSE, [0, 1, 2], []),
        (ACLOSE, [0], [1, 2]),
        (NEXT_UNTIL_TIMEOUT_THEN_CLOSE, [0], [1, 2]),
    ],
)
@pytest.mark.parametrize("as_completed", [False, True])
def test_map_termination_with_concurrency(
    termination: TerminationCase,
    completed: List[int],
    cancelled: List[int],
    as_completed: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monitored_identity = MonitoredIdentity[int]()
    s = stream(range(4)).map(
        monitored_identity.func(termination.itype),
        concurrency=2,
        as_completed=as_completed,
    )
    termination.terminate_without_leaks(s, n_pulls=1)
    assert_nothing_in_flight(monitored_identity)
    assert sorted(monitored_identity.completed) == completed
    assert monitored_identity.cancelled == cancelled
    # no error surfaced from the futures' done callbacks
    assert not caplog.records


@pytest.mark.parametrize("termination", TERMINATIONS)
def test_source_termination(termination: TerminationCase) -> None:
    # the source is closed if the client provided it as an iterator
    iterable = create_monitored_int_iterable(termination.itype, n_yields=2)
    termination.terminate_without_leaks(stream(iterable), n_pulls=1)
    assert [iterator.close_attempted for iterator in iterable.iterators] == [True]

    # ... but not closed if it is passed as an Iterable
    iterator = create_monitored_int_iterator(termination.itype, n_yields=2)
    termination.terminate_without_leaks(stream(iterator), n_pulls=1)
    assert not iterator.close_attempted


@pytest.mark.parametrize("itype", ITERABLE_TYPES)
def test_owned_source_close_error_is_raised(itype: IterableType) -> None:
    it = aiter_or_iter(
        stream(create_close_failing_monitored_int_iterable(itype)), itype
    )
    assert anext_or_next(it, itype) == 0

    # the TestError is raised on close
    with pytest.raises(TestError):
        aclose_or_close(it, itype)

    # the TestError is raised on exhaustion via __call__ / __await__
    with pytest.raises(TestError):
        await_or_call(
            stream(create_close_failing_monitored_int_iterable(itype, n_yields=2)),
            itype,
        )

    # but not raised on client-controlled exhaustion
    assert alist_or_list(
        stream(create_close_failing_monitored_int_iterable(itype, n_yields=2)), itype
    ) == [0, 1]


@pytest.mark.parametrize("itype", ITERABLE_TYPES)
def test_owned_source_closed_once_on_upstream_stop(itype: IterableType) -> None:
    iterable = create_monitored_int_iterable(itype)
    s = stream(iterable).take(1).map(identity).filter(lambda _: True).do(nothing)
    it = aiter_or_iter(s, itype)
    assert alist_or_list(it, itype) == [0]
    aclose_or_close(it, itype)
    assert [iterator.n_close_attempts for iterator in iterable.iterators] == [1]

    # only one close attempt even if the close fails
    failing_iterable = create_close_failing_monitored_int_iterable(itype)
    s = stream(failing_iterable).take(1).map(identity).filter(lambda _: True)
    it = aiter_or_iter(s, itype)
    assert anext_or_next(it, itype) == 0
    with pytest.raises(TestError):
        anext_or_next(it, itype)
    with pytest.raises(stopiteration_type(itype)):
        anext_or_next(it, itype)
    assert [iterator.n_close_attempts for iterator in failing_iterable.iterators] == [1]
    # further explicit closure attempts can be made though
    with pytest.raises(TestError):
        aclose_or_close(it, itype)
    assert [iterator.n_close_attempts for iterator in failing_iterable.iterators] == [2]


@pytest.mark.parametrize(
    "get_stream",
    [
        lambda src: stream(src).map(throw_func(TestError)),
        lambda src: stream(src).map(async_throw_func(TestError), concurrency=2),
        lambda src: stream(src).map(throw_func(TestError), concurrency=2),
        lambda src: stream(src).buffer(2).map(throw_func(TestError)),
    ],
)
@pytest.mark.parametrize("itype", ITERABLE_TYPES)
def test_await_or_call_closes_owned_source_on_error(
    itype: IterableType,
    get_stream: Callable[[Union[Iterable, AsyncIterable]], "stream[int]"],
) -> None:
    iterable = create_monitored_int_iterable(itype)
    with pytest.raises(TestError):
        await_or_call(get_stream(iterable), itype)
    assert not asyncio.all_tasks(TEST_LOOP)
    assert [iterator.n_close_attempts for iterator in iterable.iterators] == [1]


@pytest.mark.asyncio
async def test_break_cancels_pending_child_tasks() -> None:
    concurrency = 4
    monitored_identity = MonitoredIdentity[int](n_fast=1)
    async for _ in stream(range(10)).do(
        monitored_identity.async_func, concurrency=concurrency
    ):
        break
    # iterator is refcount collected after break
    await asyncio.sleep(SLOW / 10)
    assert sorted(monitored_identity.started) == [0, 1, 2, 3, 4]
    assert sorted(monitored_identity.completed) == [0]
    assert sorted(monitored_identity.cancelled) == [1, 2, 3, 4]


@pytest.mark.parametrize(
    "termination, completed, cancelled",
    [
        (EXHAUST, [0, 1, 2, 3], []),
        (AEXHAUST, [0, 1, 2, 3], []),
        (CLOSE, [0, 1], []),
        (ACLOSE, [0], [1]),
        (NEXT_UNTIL_TIMEOUT_THEN_CLOSE, [0], [1]),
    ],
)
@pytest.mark.parametrize(
    "get_stream",
    [
        lambda src: stream(src).buffer(1),
        lambda src: stream(src).group(within=timedelta(seconds=SLOW / 5)),
    ],
)
def test_puller_based_operation_termination(
    get_stream: Callable[[Union[Iterable, AsyncIterable]], "stream[int]"],
    termination: TerminationCase,
    completed: list[int],
    cancelled: list[int],
) -> None:
    src = create_monitored_int_iterator(termination.itype, n_yields=4)
    termination.terminate_without_leaks(get_stream(src), n_pulls=1)
    assert_nothing_in_flight(src)
    assert not src.close_attempted
    assert src.completed == completed
    assert src.cancelled == cancelled


@pytest.mark.parametrize("itype", ITERABLE_TYPES)
@pytest.mark.parametrize("concurrency", [1, 2])
def test_flatten_with_non_closable_owned_inner_iters(
    itype: IterableType,
    concurrency: int,
) -> None:
    src = [aiterable_or_iterable([0, 1], itype)]
    s = stream(src).flatten(concurrency=concurrency)
    it = aiter_or_iter(s, itype)
    assert anext_or_next(it, itype) == 0
    aclose_or_close(it, itype)


@pytest.mark.parametrize("itype", ITERABLE_TYPES)
@pytest.mark.parametrize("shutdown_before_close", [False, True])
def test_flatten_async_concurrent_with_sync_closable_owned_inner_iters(
    itype: IterableType,
    shutdown_before_close: bool,
) -> None:
    executor = ThreadPoolExecutor(2)
    inner = create_monitored_int_iterable("sync")
    src = [inner]
    s = stream(src).flatten(concurrency=executor)
    it = aiter_or_iter(s, itype)
    assert anext_or_next(it, itype) == 0
    if shutdown_before_close:
        executor.shutdown()
    # sync owned inner iter's closure happens properly
    # whether or not the executor has been shut down
    aclose_or_close(it, itype)
    if not shutdown_before_close:
        assert not executor._shutdown
        executor.shutdown()
    assert [it.close_attempted for it in inner.iterators] == [True]


@pytest.mark.parametrize(
    "termination, completed, cancelled, third_inner_started",
    [
        (EXHAUST, [0, 1, 2, 3], [], [0, 1, 2, 3]),
        (AEXHAUST, [0, 1, 2, 3], [], [0, 1, 2, 3]),
        (CLOSE, [0, 1], [], []),
        (ACLOSE, [0], [1], []),
        (NEXT_UNTIL_TIMEOUT_THEN_CLOSE, [0], [1], []),
    ],
)
def test_flatten_provided_inner_iter_termination(
    termination: TerminationCase,
    completed: list[int],
    cancelled: list[int],
    third_inner_started: list[int],
) -> None:
    a, b, c = [
        create_monitored_int_iterator(termination.itype, n_yields=4) for _ in range(3)
    ]
    s = stream([a, b, c]).flatten(concurrency=2)
    termination.terminate_without_leaks(s, n_pulls=2)
    # the in-flight pulls are cancelled (async) or awaited (sync), the inner iterators are not closed
    for src in (a, b):
        assert_nothing_in_flight(src)
        assert not src.close_attempted
        assert src.completed == completed
        assert src.cancelled == cancelled
    assert c.started == third_inner_started


@pytest.mark.parametrize(
    "termination, completed, cancelled, third_inner_started, closes",
    [
        (EXHAUST, [0, 1, 2, 3], [], [0, 1, 2, 3], False),
        (AEXHAUST, [0, 1, 2, 3], [], [0, 1, 2, 3], False),
        (CLOSE, [0, 1], [], [], True),
        (ACLOSE, [0], [1], [], True),
        (NEXT_UNTIL_TIMEOUT_THEN_CLOSE, [0], [1], [], True),
    ],
)
def test_flatten_owned_inner_iter_termination(
    termination: TerminationCase,
    completed: list[int],
    cancelled: list[int],
    third_inner_started: list[int],
    closes: bool,
) -> None:
    a, b, c = [
        create_monitored_int_iterable(termination.itype, n_yields=4) for _ in range(3)
    ]
    s = stream([a, b, c]).flatten(concurrency=2)
    termination.terminate_without_leaks(s, n_pulls=2)
    # the in-flight pulls are cancelled (async) or awaited (sync), the inner iterators are not closed
    for src in (a, b):
        assert len(src.iterators) == 1
        owned_iterator = src.iterators[0]
        assert_nothing_in_flight(owned_iterator)
        assert owned_iterator.close_attempted is closes
        assert owned_iterator.completed == completed
        assert owned_iterator.cancelled == cancelled
    if third_inner_started:
        assert owned_iterator.started == third_inner_started


@pytest.mark.parametrize("itype", reversed(ITERABLE_TYPES))
@pytest.mark.parametrize("concurrency", [1, 2])
def test_flatten_owned_inner_iter_failed_close_retries(
    itype: IterableType, concurrency: int
) -> None:
    iterable = create_close_failing_monitored_int_iterable(itype, n_close_failures=2)
    s = stream([iterable]).flatten(concurrency=concurrency)
    it = aiter_or_iter(s, itype)
    assert anext_or_next(it, itype) == 0
    with pytest.raises(TestError):
        aclose_or_close(it, itype)
    with pytest.raises(TestError):
        aclose_or_close(it, itype)
    aclose_or_close(it, itype)
    assert iterable.iterators[-1].n_close_attempts == 3


@pytest.mark.asyncio
async def test_flatten_cancel_as_owned_inner_pull_completes() -> None:
    consumer: asyncio.Task
    src = create_monitored_int_iterable("async")
    self_cancelling = (
        stream(src)
        # Triggers a cancellation of `await s` landing just after the pull completed
        .do(lambda _: asyncio.get_running_loop().call_soon(consumer.cancel))
    )
    consumer = asyncio.ensure_future(stream([self_cancelling]).flatten(concurrency=2))
    # the cancellation is surfaced
    with pytest.raises(asyncio.CancelledError):
        await consumer
    assert_only_current_task_is_alive()
    assert [iterator.close_attempted for iterator in src.iterators] == [True]


@pytest.mark.parametrize("termination", TERMINATIONS)
def test_add_close(termination: TerminationCase) -> None:
    termination.terminate_without_leaks(
        stream(create_monitored_int_iterator(termination.itype, n_yields=3)).buffer(1)
        + [3]
        + [4, 5],
        n_pulls=1,
    )


@pytest.mark.parametrize("termination", TERMINATIONS)
def test_observe_within_termination(termination: TerminationCase) -> None:
    monitored_identity = MonitoredIdentity[Observation](n_fast=0)
    s = stream(create_monitored_int_iterator(termination.itype, n_yields=3)).observe(
        every=timedelta(seconds=SLOW),
        do=monitored_identity.func(termination.itype),
    )
    termination.terminate_without_leaks(s, n_pulls=1)
    # the in-flight observation is awaited, even in case of cancellation
    assert monitored_identity.completed == monitored_identity.started != []


@pytest.mark.parametrize("termination", TERMINATIONS)
def test_chain_cancellation(termination: TerminationCase) -> None:
    src = create_monitored_int_iterator(termination.itype, n_fast=20, n_yields=21)
    termination.terminate_without_leaks(chain(src, itype=termination.itype), n_pulls=1)
    assert_nothing_in_flight(src)
    assert not src.close_attempted
    if termination is NEXT_UNTIL_TIMEOUT_THEN_CLOSE:
        assert len(src.cancelled) == 1


@pytest.mark.parametrize("termination", TERMINATIONS)
@pytest.mark.parametrize("n_pulls", [0, 1])
def test_chain_with_owned_source_termination(
    termination: TerminationCase, n_pulls: int
) -> None:
    iterable = create_monitored_int_iterable(termination.itype, n_fast=20, n_yields=21)
    termination.terminate_without_leaks(
        chain(iterable, itype=termination.itype),
        n_pulls=n_pulls,
    )
    assert [
        (iterator.n_close_attempts, iterator.n_pulls_in_flight_at_close)
        for iterator in iterable.iterators
    ] == [(1, 0)]


@pytest.mark.asyncio
async def test_cancel_during_cancel() -> None:
    src = SlowToCancelMonitoredIntAsyncIterator()
    closing = asyncio.ensure_future(
        ACLOSE.aterminate_without_leaks(stream(src).buffer(1), n_pulls=1)
    )
    await asyncio.sleep(SLOW / 10)
    # the close cancels the in-flight pull
    assert src.cancelled == [1] and not closing.done()
    # a second cancellation is surfaced
    closing.cancel()
    audit_closing = await audit_awaitable(closing, assert_raises=asyncio.CancelledError)
    assert audit_closing.duration < SLOW / 10
    # the interrupted close leaves the source finishing its cancellation
    await asyncio.wait(asyncio.all_tasks() - {asyncio.current_task()})
    assert_only_current_task_is_alive()


###########
# THREADS #
###########


THREAD_BACKED_STREAM_GETTERS: List[Callable[[MonitoredIdentity], "stream[int]"]] = [
    lambda monitored_identity: stream(range(9)).map(
        monitored_identity.sync_func, concurrency=2
    ),
    lambda monitored_identity: stream(range(9)).map(
        monitored_identity.sync_func, concurrency=2, as_completed=True
    ),
    lambda monitored_identity: stream[Iterator[int]](
        [map(monitored_identity.sync_func, range(9)) for _ in range(2)]
    ).flatten(concurrency=2),
]


@pytest.mark.parametrize("get_stream", THREAD_BACKED_STREAM_GETTERS)
@pytest.mark.parametrize("termination", TERMINATIONS)
def test_close_executor_backed(
    get_stream: Callable[[MonitoredIdentity[int]], "stream[int]"],
    termination: TerminationCase,
) -> None:
    monitored_identity = MonitoredIdentity[int](n_fast=0)
    s = get_stream(monitored_identity)
    if termination.itype == "async":
        stall = TEST_LOOP.run_until_complete(
            max_event_loop_stall(
                cast(AsyncTerminationCase, termination).aterminate_without_leaks(
                    s, n_pulls=1
                )
            )
        )
        # the in-flight calls are awaited without blocking the event loop
        assert stall < SLOW / 5
    else:
        termination.terminate_without_leaks(s, n_pulls=1)
    assert sorted(monitored_identity.completed) == sorted(monitored_identity.started)


@pytest.mark.asyncio
async def test_close_executor_backed_map_is_interruptible() -> None:
    monitored_identity = MonitoredIdentity[int]()
    threads = set(threading.enumerate())
    it = (
        stream(create_monitored_int_iterable("sync"))
        .map(monitored_identity.sync_func, concurrency=2)
        .__aiter__()
    )
    assert await it.__anext__() == 0
    await asyncio.sleep(SLOW / 10)
    closing = asyncio.ensure_future(it.aclose())
    await asyncio.sleep(0)
    # cancelling the close interrupts it right away
    closing.cancel()
    audit_closing = await audit_awaitable(closing, assert_raises=asyncio.CancelledError)
    assert audit_closing.duration < SLOW / 10
    # a later close waits for the in-flight calls and joins the threads
    await it.aclose()
    assert sorted(monitored_identity.completed) == sorted(monitored_identity.started)
    assert_no_new_threads(threads)
