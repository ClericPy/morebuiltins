"Copy and paste to somewhere it should be"

from typing import TypedDict


class EventTemplate(TypedDict):
    name: str  # event type, like 'user.login'
    id: str  # event id, such as type + timestamp + random
    prev_id: str  # previous(parent) event id, for event chain
    root_id: str  # origin event id, the first event id in chain
    version: str  # event version, optional
    creator: str  # event creator, client name or ip or user name
    create_at: int  # event create time, secs or ms since epoch
    expire_at: int  # event expire time, 0 for never expire
    payload: dict  # event payload data
