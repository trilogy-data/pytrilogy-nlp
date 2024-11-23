from collections import Counter

from trilogy_nlp.enums import EventType


class EventTracker:
    etype = EventType

    def __init__(self):
        self.events: Counter[EventType] = Counter()

    def count(self, event_type: EventType):
        self.events[event_type] += 1
