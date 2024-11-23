from collections import Counter

from trilogy_nlp.enums import EventType


class EventTracker:

    def __init__(self):
        self.events = Counter()

    def track(self, event_type: EventType):
        self.events[event_type] += 1
