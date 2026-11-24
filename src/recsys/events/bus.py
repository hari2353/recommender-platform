from __future__ import annotations

import json
import threading
from typing import Callable


class Bus:
    def publish(self, event_dict: dict) -> None:
        raise NotImplementedError

    def subscribe(self, handler: Callable[[dict], None]) -> None:
        raise NotImplementedError


class InMemoryBus(Bus):
    def __init__(self) -> None:
        self._handlers: list[Callable[[dict], None]] = []
        self._lock = threading.Lock()

    def publish(self, event_dict: dict) -> None:
        with self._lock:
            handlers = list(self._handlers)
        for handler in handlers:
            handler(event_dict)

    def subscribe(self, handler: Callable[[dict], None]) -> None:
        with self._lock:
            self._handlers.append(handler)

    def replay_file(self, path, limit: int | None = None) -> int:
        count = 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if limit is not None and count >= limit:
                    break
                self.publish(json.loads(line))
                count += 1
        return count


class PulsarBus(Bus):
    def __init__(
        self, url: str, topic: str, subscription: str = "recsys-processor"
    ) -> None:
        import pulsar

        self._client = pulsar.Client(url)
        self._producer = self._client.create_producer(topic)
        self._topic = topic
        self._subscription = subscription

    def publish(self, event_dict: dict) -> None:
        self._producer.send(json.dumps(event_dict).encode("utf-8"))

    def subscribe(self, handler: Callable[[dict], None]) -> None:
        import threading

        consumer = self._client.subscribe(self._topic, self._subscription)

        def _loop() -> None:
            while True:
                msg = consumer.receive()
                try:
                    handler(json.loads(msg.data().decode("utf-8")))
                    consumer.acknowledge(msg)
                except Exception:
                    consumer.negative_acknowledge(msg)

        threading.Thread(target=_loop, daemon=True).start()

    def close(self) -> None:
        self._client.close()


def get_bus(backend: str = "memory", **kwargs) -> Bus:
    if backend == "pulsar":
        return PulsarBus(
            kwargs.get("url", "pulsar://localhost:6650"),
            kwargs.get("topic", "persistent://public/default/user-events"),
        )
    return InMemoryBus()
