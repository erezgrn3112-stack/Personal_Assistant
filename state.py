from typing import Annotated, List, Optional, Dict
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

    # User preference slots (optional; populated by agent node or tool parsing)
    person_name: Optional[str]
    price_preference: Optional[str]
    location_preference: Optional[str]
    cuisine_preference: Optional[str]
    invite_someone: Optional[bool]

    # Workflow slots
    presented_options: Optional[List[Dict]]
    selected_restaurant: Optional[Dict]
    contact_phone: Optional[str]
    whatsapp_sent: Optional[bool]

    # Booking slots
    booking_day: Optional[str]
    booking_hour: Optional[str]
    booking_success: Optional[bool]
