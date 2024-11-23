from trilogy_nlp.enums import EventType
from collections import Counter

class EventTracker:

    def __init__(self):
        self.events = Counter()

    def track(self, event_type:EventType):
        self.events[event_type] += 1