import csv
import json
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from twilio.rest import Client

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

mcp = FastMCP("Restaurant-Assistant-Server")

# Hebrew CSV column names
COL_NAME = "שם"
COL_PHONE = "מספר טלפון"
COL_CUISINE = "קטגוריה"
COL_PRICE = "טווח מחירים"
COL_CITY = "עיר"
COL_DAY = "יום"
COL_HOUR = "שעה"
COL_STATUS = "סטטוס"
STATUS_FREE = "פנוי"
STATUS_BUSY = "תפוס"
SCHEDULE_DB = ROOT / "schedule_db.csv"

PRICE_ALIASES = {
    "low": "$",
    "budget": "$",
    "cheap": "$",
    "זול": "$",
    "medium": "$$",
    "mid": "$$",
    "בינוני": "$$",
    "high": "$$$",
    "premium": "$$$",
    "expensive": "$$$",
    "יוקרתי": "$$$",
    "$": "$",
    "$$": "$$",
    "$$$": "$$$",
}


def safe_read_csv(file_path: Path) -> list:
    """Read a CSV file with multi-encoding support for Hebrew text."""
    encodings = ["utf-8-sig", "cp1255", "utf-8"]
    for enc in encodings:
        try:
            with open(file_path, mode="r", encoding=enc) as f:
                return list(csv.DictReader(f))
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise Exception(f"Cannot read {file_path} — Hebrew encoding issue.")


def normalize_price(price_preference: str) -> str:
    key = price_preference.strip().lower()
    return PRICE_ALIASES.get(key, price_preference.strip())


def format_phone_for_whatsapp(phone_str: str) -> str:
    """Convert a local Israeli phone number to Twilio WhatsApp format."""
    clean_phone = "".join(filter(str.isdigit, phone_str))
    if clean_phone.startswith("0"):
        clean_phone = "972" + clean_phone[1:]
    return f"whatsapp:+{clean_phone}"


def _normalize_schedule_row(row: dict) -> dict:
    """Strip CSV keys and values so schedule rows match Hebrew column constants."""
    return {
        k.strip(): (v.strip() if isinstance(v, str) else v)
        for k, v in row.items()
    }


def _load_schedule_rows() -> list:
    """Load schedule_db.csv rows with normalized keys and values."""
    return [_normalize_schedule_row(row) for row in safe_read_csv(SCHEDULE_DB)]


def _write_schedule_rows(rows: list) -> None:
    """Persist schedule rows to schedule_db.csv with Hebrew-safe encoding."""
    fieldnames = [COL_DAY, COL_HOUR, COL_STATUS]
    with open(SCHEDULE_DB, mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@mcp.tool()
def search_restaurants(
    price_preference: str,
    location_preference: Optional[str] = None,
    cuisine_preference: Optional[str] = None,
) -> str:
    """Search restaurants_db.csv by price ($/$$/$$$), city, and cuisine."""
    try:
        restaurants = safe_read_csv(ROOT / "restaurants_db.csv")
        price_token = normalize_price(price_preference)
        matched = []

        for res in restaurants:
            if price_token not in res.get(COL_PRICE, ""):
                continue
            if location_preference and location_preference.strip() not in res.get(COL_CITY, ""):
                continue
            if cuisine_preference and cuisine_preference.strip() not in res.get(COL_CUISINE, ""):
                continue

            matched.append({
                "name": res.get(COL_NAME),
                "cuisine": res.get(COL_CUISINE),
                "price": res.get(COL_PRICE),
                "location": res.get(COL_CITY),
            })
            if len(matched) == 3:
                break

        if not matched:
            return json.dumps(
                {
                    "status": "no_results",
                    "message": "לא נמצאו מסעדות המתאימות לקריטריונים הנבחרים.",
                },
                ensure_ascii=False,
            )
        return json.dumps({"status": "success", "data": matched}, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)


@mcp.tool()
def search_contact(person_name: str) -> str:
    """Search contacts_db.csv by Hebrew name."""
    try:
        contacts = safe_read_csv(ROOT / "contacts_db.csv")

        for contact in contacts:
            if person_name.strip() in contact.get(COL_NAME, ""):
                return json.dumps(
                    {
                        "status": "success",
                        "name": contact.get(COL_NAME),
                        "phone": contact.get(COL_PHONE),
                    },
                    ensure_ascii=False,
                )

        return json.dumps(
            {"status": "not_found", "message": f"לא נמצא איש קשר בשם {person_name}."},
            ensure_ascii=False,
        )

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)


@mcp.tool()
def check_table_availability(day: str, hour: str) -> str:
    """Check whether a table slot is free in schedule_db.csv; return alternatives if busy."""
    try:
        rows = _load_schedule_rows()
        target_day = day.strip()
        target_hour = hour.strip()

        for row in rows:
            if row.get(COL_DAY) != target_day or row.get(COL_HOUR) != target_hour:
                continue

            status = row.get(COL_STATUS, "")
            if status == STATUS_FREE:
                return json.dumps(
                    {"status": "free", "message": "available"},
                    ensure_ascii=False,
                )
            if status == STATUS_BUSY:
                alternatives = [
                    r.get(COL_HOUR)
                    for r in rows
                    if r.get(COL_DAY) == target_day
                    and r.get(COL_STATUS) == STATUS_FREE
                    and r.get(COL_HOUR) != target_hour
                ][:3]
                return json.dumps(
                    {"status": "busy", "alternatives": alternatives},
                    ensure_ascii=False,
                )

        return json.dumps(
            {
                "status": "not_found",
                "message": f"לא נמצא מועד עבור {target_day} בשעה {target_hour}.",
            },
            ensure_ascii=False,
        )

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)


@mcp.tool()
def book_table_slot(day: str, hour: str) -> str:
    """Reserve a table slot by marking it busy in schedule_db.csv."""
    try:
        rows = _load_schedule_rows()
        target_day = day.strip()
        target_hour = hour.strip()
        updated = False

        for row in rows:
            if row.get(COL_DAY) != target_day or row.get(COL_HOUR) != target_hour:
                continue

            if row.get(COL_STATUS) == STATUS_BUSY:
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"השולחן ביום {target_day} בשעה {target_hour} כבר תפוס.",
                    },
                    ensure_ascii=False,
                )

            if row.get(COL_STATUS) == STATUS_FREE:
                row[COL_STATUS] = STATUS_BUSY
                updated = True
                break

        if not updated:
            return json.dumps(
                {
                    "status": "not_found",
                    "message": f"לא נמצא מועד עבור {target_day} בשעה {target_hour}.",
                },
                ensure_ascii=False,
            )

        _write_schedule_rows(rows)
        return json.dumps(
            {
                "status": "success",
                "booked": True,
                "day": target_day,
                "hour": target_hour,
            },
            ensure_ascii=False,
        )

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)


# --- Global Twilio Client Initialization (Place this near your env loading area) ---
# Initializing the client globally caches the connection pool and significantly reduces latency.
_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
FROM_WHATSAPP = os.getenv("TWILIO_WHATSAPP_NUMBER")

# Safe initialization in case keys are missing during environment setup
twilio_client = Client(_ACCOUNT_SID, _AUTH_TOKEN) if (_ACCOUNT_SID and _AUTH_TOKEN) else None


@mcp.tool()
def send_whatsapp_invitation(
    to_phone: str,
    restaurant_name: str,
    location: str,
    recipient_name: str,
    booking_day: Optional[str] = None,
    booking_hour: Optional[str] = None,
) -> str:
    """Send a personalized WhatsApp invitation via Twilio."""
    try:
        # Check against our globally cached configuration instances
        if not twilio_client or not FROM_WHATSAPP:
            return json.dumps(
                {"status": "error", "message": "Twilio credentials missing in .env"}
            )

        formatted_to_phone = format_phone_for_whatsapp(to_phone)

        # Personalized message body using the exact logic provided
        if booking_day and booking_hour:
            message_body = (
                f"היי {recipient_name.strip()}! רציתי להזמין אותך לאכול איתי במסעדת '{restaurant_name}' "
                f"ב{location} ביום {booking_day} בשעה {booking_hour}. זורם?"
            )
        else:
            message_body = (
                f"היי {recipient_name.strip()}! רציתי להזמין אותך לאכול איתי במסעדת '{restaurant_name}' "
                f"ב{location}. זורם?"
            )

        # Execute payload transmission using the persistent cached pipeline
        message = twilio_client.messages.create(
            body=message_body,
            from_=FROM_WHATSAPP,
            to=formatted_to_phone,
        )
        return json.dumps({"status": "success", "sid": message.sid}, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)
if __name__ == "__main__":
    mcp.run()
