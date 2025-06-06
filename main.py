import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Header, Request, Query as FastAPIQuery
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime, timedelta, timezone # Added timezone
import secrets # For generating secure random tokens

from supabase import create_client, Client
from postgrest.exceptions import APIError
from gotrue.errors import AuthApiError, AuthUnknownError # Added AuthUnknownError

from cryptography.fernet import Fernet
import openai

# For TF-IDF Fallback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# For Sending Email (using SendGrid as an example)
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, From, To, Subject, PlainTextContent, HtmlContent, Header as SendGridHeader

# Load environment variables from .env file
load_dotenv()

#--- Environment Variables & Supabase Client Initialization ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
# Frontend URL needed for invitation links
FRONTEND_URL_FOR_EMAILS = os.getenv("FRONTEND_URL_FOR_EMAILS", "http://localhost:3000")


if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise Exception("Supabase environment variables are not set!")

supabase_admin: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

#--- SendGrid API Key (for sending agent replies) ---
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
SENDER_EMAIL_ADDRESS = os.getenv("SENDER_EMAIL_ADDRESS") # The email address verified with SendGrid


#--- Encryption Helper Functions ---
def generate_encryption_key() -> bytes:
    """Generates a new Fernet key."""
    return Fernet.generate_key()

def encrypt_data(data: str, key: bytes) -> str:
    """Encrypts data using the provided key."""
    f = Fernet(key)
    return f.encrypt(data.encode()).decode()

def decrypt_data(encrypted_data: str, key: bytes) -> str:
    """Decrypts data using the provided key."""
    f = Fernet(key)
    return f.decrypt(encrypted_data.encode()).decode()

#--- Pydantic Models ---
class AuthenticatedUser(BaseModel):
    id: UUID
    email: Optional[EmailStr] = None
    organization_id: Optional[UUID] = None

class QueryRequest(BaseModel):
    text: str

class RAGResponse(BaseModel):
    generated_response: str
    retrieved_context_count: int
    source: str # 'openai_rag', 'openai_rag_no_context', 'tfidf_retrieval', 'tfidf_retrieval_no_match', 'tfidf_retrieval_failed', 'tfidf_failed', 'no_kb_content'

class KBContentIngestRequest(BaseModel):
    content: str

class KBEntry(BaseModel): # For responses, especially from GET /api/knowledge-base
    id: UUID # Changed from Optional[str] to UUID
    content: str
    organization_id: UUID # Changed from Optional[UUID]
    # embedding: Optional[List[float]] = None # Embedding not usually sent to frontend list
    created_at: datetime # Changed from Optional[Any]

class APISettings(BaseModel):
    openai_api_key: str

class OrganizationCreateRequest(BaseModel):
    name: str

class OrganizationResponse(BaseModel):
    id: UUID
    name: str
    created_at: datetime # Changed from Any

class InboundEmailPayload(BaseModel): # Simplified for now
    to_email: str = Field(..., alias="to") # Who the email was addressed to (our unique address)
    from_email: str = Field(..., alias="from") # Original sender
    subject: Optional[str] = None
    text: Optional[str] = None
    # For threading - names and exact location depend on email service (e.g., SendGrid payload)
    # These are common header names, often nested in a 'headers' dict in the raw payload
    message_id: Optional[str] = Field(None, alias="Message-ID") # Standard header name
    in_reply_to: Optional[str] = Field(None, alias="In-Reply-To") # Standard header name
    references: Optional[str] = None # Standard header name

class CustomerQueryDB(BaseModel): # For validating data from our DB
    id: UUID
    organization_id: UUID
    channel: str
    sender_identifier: str
    sender_name: Optional[str] = None
    subject: Optional[str] = None
    body_text: str # Should not be optional if it's the primary content
    status: str
    received_at: datetime # Changed from Any
    original_created_at: Optional[datetime] = None # Changed from Any
    # Fields for AI draft if saved with the query
    ai_draft_response: Optional[str] = None
    ai_response_source: Optional[str] = None
    ai_retrieved_context_count: Optional[int] = None
    updated_at: Optional[datetime] = None # Changed from Any

class QueryMessageDB(BaseModel):
    id: UUID
    customer_query_id: UUID
    organization_id: UUID # Denormalized for easier direct queries on messages if needed
    sender_type: str  # 'customer', 'agent', 'ai_draft', 'system_note'
    sender_identifier: Optional[str] = None # customer's email or agent's user_id
    body_text: str
    message_id_header: Optional[str] = None # From original email if applicable
    in_reply_to_header: Optional[str] = None # From original email if applicable
    created_at: datetime # Changed from Any

class ProcessQueryRequest(BaseModel):
    # Making Al fields optional as not all status updates involve an AI draft
    ai_draft_response: Optional[str] = None
    ai_response_source: Optional[str] = None
    ai_retrieved_context_count: Optional[int] = None
    new_status: str

class AgentReplyRequest(BaseModel):
    reply_text: str

class DashboardStats(BaseModel):
    total_queries: int
    new_queries: int
    agent_replied_queries: int
    customer_reply_queries: int
    closed_queries: int

class QueryVolumeDataPoint(BaseModel):
    date: str  # YYYY-MM-DD
    query_count: int

class QueryVolumeResponse(BaseModel):
    data: List[QueryVolumeDataPoint]
    period_days: int

# --- Invitation Models ---
class InvitationCreateRequest(BaseModel):
    email: EmailStr
    role: str = Field("agent", pattern="^(agent|admin)$") # Example roles

class InvitationDB(BaseModel): # For database interaction and response
    id: UUID
    organization_id: UUID
    email: EmailStr
    role: str
    token: str
    status: str
    created_at: datetime
    expires_at: datetime
    invited_by_user_id: Optional[UUID] = None

class AcceptInvitationRequest(BaseModel):
    token: str

class OrganizationMember(BaseModel): # For listing current members
    id: UUID # This is the profile_id / user_id of the member
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None
    # role: str # TODO: Add role when role management is in profiles or org_users table
    joined_at: Optional[datetime] = None # This is essentially profiles.created_at

class PendingInvitation(BaseModel): # For listing pending invites
    id: UUID
    email: EmailStr
    role: str
    status: str # Should always be 'pending' for this list
    created_at: datetime
    expires_at: datetime
    invited_by_user_id: Optional[UUID] = None

#--- FastAPI App Initialization ---
app = FastAPI(title="AI Support Responder API", version="1.0.0")

#--- CORS Configuration ---
# Ensure FRONTEND_URL is set in your .env if your frontend is deployed elsewhere
# Also add FRONTEND_URL_FOR_EMAILS if it's different (e.g. if your app is at app.domain.com and emails link to just domain.com)
origins = [
    "http://localhost:3000","https://ai-support-responder-frontend.vercel.app",
    os.getenv("FRONTEND_URL_FOR_EMAILS"), # For links in emails like accept invitation
    os.getenv("FRONTEND_URL") # For general API calls from the app
]
# Filter out None values and remove duplicates if URLs are the same
origins = list(set(filter(None, origins))) # Ensures unique, non-None origins
if not origins: # Fallback if no env variables are set
    origins = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#--- Helper: Get Decrypted OpenAI API Key ---
async def get_decrypted_openai_api_key(org_id: UUID, admin_client: Client) -> Optional[str]:
    try:
        settings_response = (
            admin_client.table("organization_settings")
            .select("openai_api_key_encrypted, encryption_key")
            .eq("organization_id", str(org_id))
            .maybe_single() # Use maybe_single to gracefully handle no settings
            .execute()
        )

        if settings_response.data: # Check if data exists and is not None
            encrypted_api_key = settings_response.data.get("openai_api_key_encrypted")
            encryption_key_str = settings_response.data.get("encryption_key")
            if encrypted_api_key and encryption_key_str:
                key_bytes = encryption_key_str.encode()
                decrypted_key = decrypt_data(encrypted_api_key, key_bytes)
                return decrypted_key
        else:
            # This case means no settings row was found for the org_id, or the row was empty
            print(f"API_KEY_HELPER: No settings found for org {org_id}, or data is None/empty.")
            
    except APIError as e: # Catch PostgREST errors specifically
        print(f"API_KEY_HELPER: Supabase APIError fetching API key for org {org_id}: {e}")
    except Exception as e: # Catch other errors like decryption issues
        print(f"API_KEY_HELPER: Decryption or other error fetching API key for org {org_id}: {e}")
    return None


#--- Dependency for getting the current user from JWT ---
async def get_current_user_dependency(authorization: str = Header(None)):
    # This dependency gets user auth data. Org association check is separate.
    if not authorization or not authorization.startswith("Bearer "):
        print("AUTH_DEBUG: Missing or invalid Authorization header")
        raise HTTPException(status_code=401, detail="Not authenticated: Missing or invalid token")
    
    token = authorization.split(" ")[1]
    print(f"AUTH_DEBUG: Received token: {token[:20]}...")
    
    user_id_from_token: Optional[UUID] = None
    user_email_from_token: Optional[EmailStr] = None
    org_id_from_profile: Optional[UUID] = None

    try:
        # Validate token and get user details from Supabase Auth
        user_response = supabase_admin.auth.get_user(token)
        print(f"AUTH_DEBUG: supabase_admin.auth.get_user response: {user_response}")

        if not user_response: # More robust check for None response object
            print("AUTH_DEBUG: Token validation failed: get_user() returned None or a falsy value.")
            raise HTTPException(status_code=401, detail="Invalid token: Authentication service error (no response object)")

        authed_user = user_response.user # Assuming user_response itself is not None
        print(f"AUTH_DEBUG: user_response.user: {authed_user}")

        if not authed_user: # Check if the user object within the response is None
            error_detail_from_supabase = "Unknown auth error"
            # Attempt to get more specific error if Supabase provides it
            if hasattr(user_response, 'error') and user_response.error: 
                error_detail_from_supabase = str(user_response.error)
            print(f"AUTH_DEBUG: Token validation failed: No user object in response. Supabase error: {error_detail_from_supabase}. Full response: {user_response}")
            raise HTTPException(status_code=401, detail=f"Invalid or expired token (no user data found): {error_detail_from_supabase}")

        user_id_from_token = UUID(authed_user.id)
        user_email_from_token = authed_user.email # Pydantic will validate if it's EmailStr
        # This type ignore might be needed if linters complain about EmailStr direct assignment
        # user_email_from_token = EmailStr(authed_user.email) if authed_user.email else None # type: ignore

        print(f"AUTH_DEBUG: User authenticated: ID={user_id_from_token}, Email={user_email_from_token}")
        
        # Fetch profile data to get organization_id
        profile_response = (
            supabase_admin.table("profiles")
            .select("organization_id")
            .eq("id", str(user_id_from_token))
            .maybe_single() # Use maybe_single to handle 0 or 1 row gracefully
            .execute()
        )

        if profile_response is None: # Explicitly check if the response object itself is None
            print(f"AUTH_DEBUG: Profile fetch query execution returned None for user {user_id_from_token}.")
            # This is highly unusual for a .execute() call, indicates a severe issue if it happens
            raise HTTPException(status_code=500, detail="Internal server error: Failed to retrieve profile information.")

        print(f"AUTH_DEBUG: Profile fetch response object: {type(profile_response)}")
        profile_data = profile_response.data # Assign to variable for clarity
        print(f"AUTH_DEBUG: Profile fetch response data for user {user_id_from_token}: {profile_data}")

        if profile_data and profile_data.get("organization_id"):
            org_id_from_profile = UUID(profile_data["organization_id"])
            print(f"AUTH_DEBUG: Found organization_id in profile: {org_id_from_profile}")
        else:
            # Profile might exist but no org_id (e.g., new user yet to create/join org)
            # Or profile itself doesn't exist (which should be handled by signup trigger, but good to log)
            print(f"AUTH_DEBUG: No organization_id found in profile for user {user_id_from_token}. Profile data: {profile_data}")
            # For endpoints requiring an org, this user won't pass get_current_user_with_org_dependency

        return AuthenticatedUser(
            id=user_id_from_token,
            email=user_email_from_token, # type: ignore
            organization_id=org_id_from_profile
        )

    except AuthApiError as e: # Specific GoTrue/Supabase auth errors
        print(f"AUTH_DEBUG: Supabase AuthApiError during token validation: {e}")
        raise HTTPException(status_code=401, detail=f"Authentication failure: {e.message}")
    except APIError as e: # Specific PostgREST errors during profile fetch
        print(f"AUTH_DEBUG: Supabase APIError during user/profile fetch: {e}")
        raise HTTPException(status_code=500, detail=f"Database error during authentication: {e.message}")
    except Exception as e: # Catch-all for other unexpected errors
        print(f"AUTH_DEBUG: Unexpected error during authentication: {type(e).__name__} - {e}")
        error_message = str(e) if str(e) else "An unexpected authentication error occurred."
        raise HTTPException(status_code=401, detail=f"Authentication error: {error_message}")


async def get_current_user_with_org_dependency(current_user: AuthenticatedUser = Depends(get_current_user_dependency)):
    """
    Dependency that ensures the authenticated user is also associated with an organization.
    """
    if not current_user.organization_id:
        print(f"User {current_user.id} not associated with an organization, but org_id is required for this endpoint.")
        raise HTTPException(status_code=403, detail="User not associated with an organization. Please complete onboarding.")
    return current_user

#--- New Helper Function to Send Email ---
async def send_actual_email(
    to_email: str,
    subject: str,
    plain_text_content: str,
    html_content: Optional[str] = None,
    in_reply_to_header_val: Optional[str] = None,
    references_header_val: Optional[str] = None,
    from_name: Optional[str] = "Your Support Team" # Default From name
):
    if not SENDGRID_API_KEY or not SENDER_EMAIL_ADDRESS:
        print("EMAIL_SEND: SendGrid API Key or Sender Email Address is not configured. Skipping email send.")
        return False

    message = Mail(
        from_email=From(SENDER_EMAIL_ADDRESS, from_name), # Use dynamic or configurable from_name
        to_emails=To(to_email), # SendGrid's To helper can take a string or list
        subject=Subject(subject),
        plain_text_content=PlainTextContent(plain_text_content)
    )
    if html_content:
        message.html_content = HtmlContent(html_content)
    
    # Add headers for threading correctly
    if in_reply_to_header_val:
        message.add_header(SendGridHeader("In-Reply-To", in_reply_to_header_val))
        print(f"EMAIL_SEND: Added In-Reply-To header: {in_reply_to_header_val}")
    if references_header_val:
        message.add_header(SendGridHeader("References", references_header_val))
        print(f"EMAIL_SEND: Added References header: {references_header_val}")

    try:
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
        print(f"EMAIL_SEND: Email sent to {to_email}. Status Code: {response.status_code}")
        if 200 <= response.status_code < 300: # Check for successful status codes
            return True
        else:
            # Log the full SendGrid error if available
            print(f"EMAIL_SEND: SendGrid error: Status {response.status_code}, Body: {response.body}")
            return False
    except Exception as e:
        print(f"EMAIL_SEND: Failed to send email to {to_email}: {e}")
        return False

#--- API Endpoints ---
@app.get("/")
def read_root():
    return {"status": "AI Support Responder Backend is running."}

#--- Organization Management ---
@app.post("/api/organizations", response_model=OrganizationResponse, status_code=201)
async def create_organization(
    org_data: OrganizationCreateRequest,
    current_user: AuthenticatedUser = Depends(get_current_user_dependency) # Uses base dependency, org_id can be None
):
    print(f"ORG_CREATE: Attempting to create organization '{org_data.name}' for user {current_user.id}")
    if current_user.organization_id: # User already belongs to an organization
        raise HTTPException(status_code=400, detail="User is already associated with an organization.")

    new_org_uuid = uuid4() # Generate a new UUID for the organization
    new_org_payload = {"name": org_data.name, "id": str(new_org_uuid)}
    
    try:
        print(f"ORG_CREATE: Inserting into 'organizations' table: {new_org_payload}")
        new_org_response = (
            supabase_admin.table("organizations") # Ensure this table name is correct in your DB
            .insert(new_org_payload)
            .execute() # Removed .select()
        )

        # Check for errors from the insert operation
        if hasattr(new_org_response, 'error') and new_org_response.error:
            print(f"ORG_CREATE: Error inserting organization. Supabase error: {new_org_response.error}")
            raise APIError(new_org_response.error.model_dump()) # Raise APIError to be caught below

        if not new_org_response.data: # Check if data list is empty (should contain the new org)
            print(f"ORG_CREATE: Organization insert failed or returned no data. Response: {new_org_response}")
            raise APIError({"message": "Failed to create organization: No data returned from insert."})


        created_org = new_org_response.data[0]
        new_org_id = UUID(created_org["id"]) # Confirm this matches new_org_uuid
        print(f"ORG_CREATE: Organization created with ID: {new_org_id}. Now updating profile.")

        # Update the user's profile with the new organization_id
        update_profile_response = (
            supabase_admin.table("profiles")
            .update({"organization_id": str(new_org_id)})
            .eq("id", str(current_user.id))
            .execute() # Removed .select()
        )

        if hasattr(update_profile_response, 'error') and update_profile_response.error:
            print(f"ORG_CREATE: Error updating profile. Supabase error: {update_profile_response.error}")
            # Critical error: org created but profile not linked. May need manual cleanup or rollback logic.
            raise APIError(update_profile_response.error.model_dump())

        if not update_profile_response.data:
            print(f"ORG_CREATE: Profile update failed or returned no data. Response: {update_profile_response}")
            # This is also critical.
            raise APIError({"message": "Failed to link organization to user profile: No data returned from update."})
        
        print(f"ORG_CREATE: Profile for user {current_user.id} updated with organization ID {new_org_id}.")
        # Prepare data for OrganizationResponse Pydantic model
        response_data = {
            "id": new_org_id,
            "name": created_org["name"],
            "created_at": datetime.fromisoformat(created_org["created_at"]) # Ensure created_at is datetime
        }
        return OrganizationResponse(**response_data)

    except APIError as e: # Catch PostgREST errors
        print(f"ORG_CREATE: Caught APIError during organization creation/profile update: {e.json() if hasattr(e, 'json') else str(e)}")
        # Attempt to provide a more specific code if available from PostgREST error
        status_code = e.code if hasattr(e, 'code') and isinstance(e.code, int) else 500
        error_message = e.message if hasattr(e, 'message') else str(e)
        raise HTTPException(status_code=status_code, detail=f"Database error: {error_message}")
    except Exception as e: # Catch other unexpected errors
        print(f"ORG_CREATE: Unexpected error: {type(e).__name__} - {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error during organization creation: {str(e)}")

#--- Invitation Endpoints ---
@app.post("/api/organizations/invitations", response_model=InvitationDB, status_code=201)
async def create_invitation(
    invitation_data: InvitationCreateRequest,
    current_user: AuthenticatedUser = Depends(get_current_user_with_org_dependency) # Requires inviter to be in an org
):
    org_id = current_user.organization_id # Should be valid due to dependency
    invited_by_user_id = current_user.id
    invitee_email = invitation_data.email.lower() # Standardize email for checks

    if not org_id: # Defensive check, should be caught by dependency
        raise HTTPException(status_code=403, detail="Inviter must belong to an organization.")

    print(f"INVITE: User {invited_by_user_id} inviting {invitee_email} to org {org_id} with role {invitation_data.role}")

    try:
        # Check for existing PENDING invitation for this email to this organization
        pending_invite_check_response = (
            supabase_admin.table("invitations")
            .select("id", count="exact") # Only need count
            .eq("email", invitee_email)
            .eq("organization_id", str(org_id))
            .eq("status", "pending")
            .execute()
        )
        if pending_invite_check_response.count and pending_invite_check_response.count > 0:
            raise HTTPException(status_code=400, detail=f"An active invitation already exists for {invitee_email} for this organization.")

        # Check if a user with this email is already a member of THIS organization.
        # This requires joining auth.users with profiles. We can use an RPC function for this.
        # For simplicity here, we assume if `list_users` finds a user, and their profile has this org_id, they are a member.
        # This is a simplified check. A dedicated DB function would be more robust.
        
        # Corrected way to check if user with this email exists:
        existing_users_response = supabase_admin.auth.admin.list_users(page=1, per_page=10) # Fetch users; no direct email filter here in v2 of gotrue-py for admin
        
        found_existing_user_in_org = False
        if existing_users_response and existing_users_response.users:
            for user_in_auth in existing_users_response.users:
                if user_in_auth.email and user_in_auth.email.lower() == invitee_email:
                    # User with this email exists in auth.users. Now check their profile for this org.
                    profile_check = (
                        supabase_admin.table("profiles")
                        .select("id")
                        .eq("id", str(user_in_auth.id))
                        .eq("organization_id", str(org_id))
                        .maybe_single()
                        .execute()
                    )
                    if profile_check.data:
                        found_existing_user_in_org = True
                        break
        
        if found_existing_user_in_org:
            raise HTTPException(status_code=400, detail=f"{invitee_email} is already a member of this organization.")


        token = secrets.token_urlsafe(32)
        now_utc = datetime.now(timezone.utc) # Ensure timezone awareness
        expires_at = now_utc + timedelta(days=7)

        org_name = "Your Organization" # Default
        try:
            # Fetch organization name for the email content
            org_details_response = supabase_admin.table("organizations").select("name").eq("id", str(org_id)).single().execute()
            if org_details_response.data:
                org_name = org_details_response.data["name"]
        except APIError: # Catch PostgREST error
            print(f"INVITE: Could not fetch organization name for org {org_id}. Using default.")
        except Exception as e_org_name: # Catch other errors
            print(f"INVITE: Error fetching org name: {e_org_name}. Using default.")


        invitation_payload = {
            "organization_id": str(org_id),
            "email": invitee_email,
            "role": invitation_data.role,
            "token": token,
            "status": "pending",
            "created_at": now_utc.isoformat(), # Store in ISO format
            "expires_at": expires_at.isoformat(), # Store in ISO format
            "invited_by_user_id": str(invited_by_user_id)
        }
        
        insert_response = supabase_admin.table("invitations").insert(invitation_payload).execute() # Removed .select()

        if not insert_response.data: # Check for error on insert
            raise APIError({"message": "Failed to save invitation or no data returned."}) # Raise to be caught
        
        created_invitation_data = insert_response.data[0]
        # Ensure date fields are actual datetime objects for Pydantic validation if they come back as strings
        created_invitation_data['created_at'] = datetime.fromisoformat(created_invitation_data['created_at'])
        created_invitation_data['expires_at'] = datetime.fromisoformat(created_invitation_data['expires_at'])

        print(f"INVITE: Invitation saved with ID {created_invitation_data['id']}")
        
        invitation_link = f"{FRONTEND_URL_FOR_EMAILS}/accept-invitation?token={token}"
        email_subject = f"You're invited to join {org_name} on AI Responder"
        email_text_content = (
            f"Hello,\n\n"
            f"You have been invited to join the organization '{org_name}' on AI Responder "
            f"with the role of '{invitation_data.role}'.\n\n"
            f"To accept this invitation, please click on the link below:\n"
            f"{invitation_link}\n\n"
            f"This link will expire in 7 days.\n\n"
            f"If you did not expect this invitation, please ignore this email.\n\n"
            f"Thanks,\nThe AI Responder Team"
        )
        email_html_content = ( # Basic HTML for the email
            f"<p>Hello,</p>"
            f"<p>You have been invited to join the organization '<strong>{org_name}</strong>' on AI Responder "
            f"with the role of '<strong>{invitation_data.role}</strong>'.</p>"
            f"<p>To accept this invitation, please click on the link below:</p>"
            f'<p><a href="{invitation_link}">{invitation_link}</a></p>'
            f"<p>This link will expire in 7 days.</p>"
            f"<p>If you did not expect this invitation, please ignore this email.</p>"
            f"<p>Thanks,<br>The AI Responder Team</p>"
        )

        email_sent = await send_actual_email(
            to_email=invitee_email,
            subject=email_subject,
            plain_text_content=email_text_content,
            html_content=email_html_content,
            from_name=f"{org_name} (via AI Responder)" # e.g. "Client Company (via AI Responder)"
        )
        if not email_sent:
            # Even if email fails, the invite is in the DB. Admin might need to resend/notify.
            print(f"INVITE: Failed to send invitation email to {invitee_email}, but invitation record created.")
            # Depending on policy, you might raise an error here or just log it.
            # For now, we return the created invitation data.

        return InvitationDB(**created_invitation_data) # Use Pydantic model for response

    except APIError as e: # Catch PostgREST errors specifically
        print(f"INVITE: DB error during invitation process: {e.json() if hasattr(e, 'json') else str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error during invitation: {e.message if hasattr(e, 'message') else str(e)}")
    except HTTPException as e: # Re-raise HTTPExceptions from checks
        raise e
    except Exception as e: # Catch other unexpected errors
        print(f"INVITE: Unexpected error creating invitation: {type(e).__name__} - {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error creating invitation: {str(e)}")

@app.post("/api/invitations/accept", status_code=200)
async def accept_invitation(
    request_data: AcceptInvitationRequest,
    current_user: AuthenticatedUser = Depends(get_current_user_dependency) # User must be logged in
):
    token = request_data.token
    accepting_user_id = current_user.id
    accepting_user_email = current_user.email # This is EmailStr|None from AuthenticatedUser

    print(f"ACCEPT_INVITE: User {accepting_user_id} attempting to accept with token {token}")

    try:
        # 1. Find the invitation by token
        invitation_response = (
            supabase_admin.table("invitations")
            .select("*") # Select all fields to construct InvitationDB
            .eq("token", token)
            .maybe_single() # Expect one or none
            .execute()
        )

        if not invitation_response.data:
            raise HTTPException(status_code=404, detail="Invitation token not found.")

        # Convert to Pydantic model, handling date parsing
        inv_data = invitation_response.data
        inv_data['created_at'] = datetime.fromisoformat(inv_data['created_at'].replace('Z', '+00:00'))
        inv_data['expires_at'] = datetime.fromisoformat(inv_data['expires_at'].replace('Z', '+00:00'))
        invitation = InvitationDB(**inv_data)

        # 2. Validate invitation status and expiry
        if invitation.status != "pending":
            raise HTTPException(status_code=400, detail=f"Invitation is already {invitation.status}.")
        
        # Ensure expires_at is timezone-aware for comparison
        if invitation.expires_at.tzinfo is None: # Should be set by fromisoformat if Z or offset was present
            invitation.expires_at = invitation.expires_at.replace(tzinfo=timezone.utc)
        
        if datetime.now(timezone.utc) > invitation.expires_at:
            # Optionally update status to 'expired' in DB
            supabase_admin.table("invitations").update({"status": "expired"}).eq("id", str(invitation.id)).execute()
            raise HTTPException(status_code=400, detail="Invitation has expired.")

        # 3. Verify the logged-in user's email matches the invitation email (case-insensitive)
        if not accepting_user_email: # Should not happen if get_current_user_dependency worked
             raise HTTPException(status_code=401, detail="Could not verify authenticated user's email.")
        
        if invitation.email.lower() != accepting_user_email.lower():
            raise HTTPException(status_code=403, detail="This invitation is intended for a different email address.")

        # 4. Check if the user is already part of an organization
        # The current_user object from get_current_user_dependency already has organization_id from profiles
        if current_user.organization_id:
            if current_user.organization_id == invitation.organization_id:
                # User is already part of this organization. Mark invite as accepted.
                (supabase_admin.table("invitations")
                    .update({"status": "accepted"})
                    .eq("id", str(invitation.id))
                    .execute())
                return {"message": "You are already a member of this organization. Invitation marked as accepted."}
            else:
                # User is part of a DIFFERENT organization - this is a business rule.
                # For now, we prevent joining multiple orgs.
                raise HTTPException(status_code=400, detail="You are already a member of another organization.")
        
        # 5. User is authenticated, email matches, not part of any org OR not part of this org (latter handled above). Update their profile.
        update_profile_payload = {"organization_id": str(invitation.organization_id)}
        # TODO: Add role assignment here if profiles table has a role column,
        # or create an entry in a separate organization_users table with the role.
        # e.g., update_profile_payload["role_in_org"] = invitation.role 

        update_profile_response = (
            supabase_admin.table("profiles")
            .update(update_profile_payload)
            .eq("id", str(accepting_user_id))
            .execute() # Removed .select()
        )

        if not update_profile_response.data: # Check if data (updated row) is returned.
            # This implies the update failed or returned no data.
            # PostgREST default for UPDATE is return=representation, so data should be there.
            raise APIError({"message": "Failed to update user profile with organization or no data returned."})

        # 6. Update invitation status to 'accepted'
        (
            supabase_admin.table("invitations")
            .update({"status": "accepted"})
            .eq("id", str(invitation.id))
            .execute()
        )
        
        print(f"ACCEPT_INVITE: User {accepting_user_id} successfully joined organization {invitation.organization_id}")
        return {"message": "Invitation accepted successfully! You have joined the organization."}

    except APIError as e: # Catch PostgREST errors
        print(f"ACCEPT_INVITE: Database error: {e.json() if hasattr(e, 'json') else str(e)}")
        error_detail = e.message if hasattr(e, 'message') else str(e)
        # Determine appropriate status code if possible
        status_code = e.code if hasattr(e, 'code') and isinstance(e.code, int) and e.code >= 400 else 500
        raise HTTPException(status_code=status_code, detail=f"Database error accepting invitation: {error_detail}")
    except HTTPException as e: # Re-raise our own HTTPExceptions
        raise e
    except Exception as e: # Catch other unexpected errors
        print(f"ACCEPT_INVITE: Unexpected error: {type(e).__name__} - {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error accepting invitation: {str(e)}")


@app.delete("/api/invitations/{invitation_id}", status_code=200)
async def revoke_invitation(
    invitation_id: UUID,
    current_user: AuthenticatedUser = Depends(get_current_user_with_org_dependency)
):
    org_id = current_user.organization_id
    print(f"REVOKE_INVITE: User {current_user.id} attempting to revoke invitation {invitation_id} for org {org_id}")

    try:
        # Find the invitation and verify it belongs to the user's organization and is pending
        invitation_response = (
            supabase_admin.table("invitations")
            .select("id, status")
            .eq("id", str(invitation_id))
            .eq("organization_id", str(org_id)) # Ensure it's for their org
            .maybe_single()
            .execute()
        )

        if not invitation_response.data:
            raise HTTPException(status_code=404, detail="Invitation not found or you do not have permission to modify it.")

        invitation_status = invitation_response.data.get("status")
        if invitation_status != "pending":
            raise HTTPException(status_code=400, detail=f"Cannot revoke invitation with status '{invitation_status}'. Only pending invitations can be revoked.")

        # Update invitation status to 'revoked'
        update_response = (
            supabase_admin.table("invitations")
            .update({"status": "revoked"})
            .eq("id", str(invitation_id))
            .execute() # Removed .select()
        )

        if not update_response.data: # Check if data (updated row) is returned
             # If no data and no error, it implies update was successful with return=minimal
             # or RLS masked it (though service_role bypasses RLS)
            if hasattr(update_response, 'error') and update_response.error:
                raise APIError(update_response.error.model_dump()) # Raise to be caught
            # Assume success if no error and no data (for return=minimal)
            print(f"REVOKE_INVITE: Invitation {invitation_id} status updated to revoked (no data returned by update).")
        else:
            print(f"REVOKE_INVITE: Invitation {invitation_id} status updated to revoked.")
            
        return {"message": "Invitation revoked successfully."}

    except APIError as e:
        print(f"REVOKE_INVITE: Database error: {e.json() if hasattr(e, 'json') else str(e)}")
        error_detail = e.message if hasattr(e, 'message') else str(e)
        status_code = e.code if hasattr(e, 'code') and isinstance(e.code, int) and e.code >= 400 else 500
        raise HTTPException(status_code=status_code, detail=f"Database error revoking invitation: {error_detail}")
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"REVOKE_INVITE: Unexpected error: {type(e).__name__} - {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error revoking invitation: {str(e)}")


@app.get("/api/organizations/members", response_model=List[OrganizationMember])
async def get_organization_members(
    current_user: AuthenticatedUser = Depends(get_current_user_with_org_dependency)
):
    org_id = current_user.organization_id # Ensured by dependency
    print(f"MEMBERS: Fetching members for org {org_id}")
    members_list = []
    try:
        profiles_response = (
            supabase_admin.table("profiles")
            .select("id, full_name, created_at") # created_at from profiles is when they joined/profile created
            .eq("organization_id", str(org_id))
            .execute()
        )
        if profiles_response.data:
            for profile_data_dict in profiles_response.data:
                user_email = None
                try:
                    # Fetch email from auth.users using the admin client
                    user_auth_info_response = supabase_admin.auth.admin.get_user_by_id(str(profile_data_dict["id"]))
                    if user_auth_info_response and user_auth_info_response.user:
                        user_email = user_auth_info_response.user.email
                except (AuthApiError, AuthUnknownError, Exception) as auth_error: # Catch specific auth errors
                    print(f"MEMBERS: Could not fetch email for user ID {profile_data_dict['id']}: {auth_error}")
                
                # Ensure joined_at is parsed correctly
                joined_at_val = profile_data_dict.get("created_at")
                joined_at_dt = datetime.fromisoformat(joined_at_val.replace('Z', '+00:00')) if isinstance(joined_at_val, str) else joined_at_val

                members_list.append(
                    OrganizationMember(
                        id=UUID(profile_data_dict["id"]),
                        full_name=profile_data_dict.get("full_name"),
                        email=user_email, # type: ignore # Pydantic will validate
                        joined_at=joined_at_dt
                    )
                )
        return members_list
    except APIError as e: # Catch PostgREST errors
        print(f"MEMBERS: APIError fetching members for org {org_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Database error fetching members: {e.message}")
    except Exception as e: # Catch other errors
        print(f"MEMBERS: Unexpected error fetching members for org {org_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error fetching members: {str(e)}")

@app.get("/api/organizations/invitations/pending", response_model=List[PendingInvitation])
async def get_pending_invitations(
    current_user: AuthenticatedUser = Depends(get_current_user_with_org_dependency)
):
    org_id = current_user.organization_id # Ensured by dependency
    print(f"PENDING_INVITES: Fetching pending invitations for org {org_id}")
    try:
        invitations_response = (
            supabase_admin.table("invitations")
            .select("*") # Select all fields for pending invitation display
            .eq("organization_id", str(org_id))
            .eq("status", "pending") # Only fetch pending ones
            .order("created_at", desc=True) # Show newest pending invites first
            .execute()
        )
        # Convert string dates to datetime for Pydantic validation before returning
        parsed_invitations = []
        if invitations_response.data:
            for item_dict in invitations_response.data:
                # Ensure date fields are converted correctly, handling potential 'Z' for UTC
                item_dict['created_at'] = datetime.fromisoformat(item_dict['created_at'].replace('Z', '+00:00')) if item_dict.get('created_at') else None
                item_dict['expires_at'] = datetime.fromisoformat(item_dict['expires_at'].replace('Z', '+00:00')) if item_dict.get('expires_at') else None
                if item_dict['created_at'] and item_dict['expires_at']: # Only include if dates are valid
                    parsed_invitations.append(PendingInvitation(**item_dict))
                else:
                    # Log if an invitation has invalid date format but don't stop processing others
                    print(f"Warning: Skipping invitation with invalid/missing date format: {item_dict.get('id')}")
        return parsed_invitations
    except APIError as e: # Catch PostgREST errors
        print(f"PENDING_INVITES: APIError fetching pending invitations for org {org_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Database error fetching pending invitations: {e.message}")
    except Exception as e: # Catch other errors like date parsing
        print(f"PENDING_INVITES: Unexpected error fetching pending invitations for org {org_id}: {type(e).__name__} - {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error fetching pending invitations: {str(e)}")


#--- API Key Management Endpoints ---
@app.post("/api/settings/api-key")
async def save_api_key(
    settings: APISettings,
    current_user: AuthenticatedUser = Depends(get_current_user_with_org_dependency)
):
    org_id = current_user.organization_id # Ensured by dependency
    encryption_k_bytes = generate_encryption_key()
    encrypted_api_key = encrypt_data(settings.openai_api_key, encryption_k_bytes)
    
    payload_to_upsert = {
        "organization_id": str(org_id),
        "openai_api_key_encrypted": encrypted_api_key,
        "encryption_key": encryption_k_bytes.decode(), # Store the key as string
    }
    print(f"Attempting to upsert API key for org: {org_id}")
    try:
        db_response = (
            supabase_admin.table("organization_settings")
            .upsert(payload_to_upsert, on_conflict="organization_id") # Assumes organization_id has a UNIQUE constraint
            .execute()
        )
        # For upsert, PostgREST might return empty data on successful update/insert if return=minimal
        # A successful operation might not always have data if that's the PostgREST config.
        # Check for error presence first.
        if hasattr(db_response, 'error') and db_response.error:
            error_message = db_response.error.message
            print(f"API key upsert failed for org {org_id}. Error: {error_message}")
            raise HTTPException(status_code=500, detail=f"Failed to save API key: {error_message}")

        print(f"API key upsert successful for org {org_id}. Response data: {db_response.data}")
        return {"message": "API key saved successfully"}

    except APIError as e: # Catch PostgREST specific errors
        print(f"APIError saving API key for org {org_id}: {e}")
        # Try to use PostgREST error code if available
        status_code = e.code if hasattr(e, 'code') and isinstance(e.code, int) else 500
        raise HTTPException(status_code=status_code, detail=f"Database error saving API key: {e.message}")
    except Exception as e: # Catch other errors like during encryption
        print(f"Unexpected error saving API key for org {org_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error saving API key: {str(e)}")


#--- Knowledge Base Endpoints ---
@app.post("/api/knowledge-base/ingest", response_model=KBEntry)
async def ingest_knowledge_base_content(
    request_data: KBContentIngestRequest,
    current_user: AuthenticatedUser = Depends(get_current_user_with_org_dependency)
):
    org_id = current_user.organization_id # Ensured by dependency
    content_to_ingest = request_data.content
    embedding_vector = None # Will store the generated embedding
    
    decrypted_api_key = await get_decrypted_openai_api_key(org_id, supabase_admin)

    if decrypted_api_key:
        try:
            print(f"KB_INGEST: Attempting OpenAI embedding for org {org_id}")
            openai_client = openai.OpenAI(api_key=decrypted_api_key)
            embedding_response = openai_client.embeddings.create(
                input=content_to_ingest,
                model="text-embedding-ada-002" # OpenAI's recommended embedding model
            )
            embedding_vector = embedding_response.data[0].embedding
            print(f"KB_INGEST: Successfully generated OpenAI embedding for org {org_id}")
        except openai.APIError as e: # Catch OpenAI specific API errors (quota, auth, etc.)
            print(f"KB_INGEST: OpenAI API error during ingestion for org {org_id} (non-critical, proceeding without OpenAI embedding): {e}")
            # Log the error but allow proceeding without embedding for now
        except Exception as e: # Catch other unexpected errors during OpenAI call
            print(f"KB_INGEST: Unexpected error during OpenAI call for org {org_id}: {e}")
            # Log error, proceed without embedding
    else:
        print(f"KB_INGEST: No OpenAI API key found for org {org_id}. Proceeding without OpenAI embedding.")

    # Prepare data for insertion into knowledge_base_entries
    data_to_insert = {
        "organization_id": str(org_id),
        "content": content_to_ingest,
        "embedding": embedding_vector # This will be None if OpenAI embedding failed or no key
    }
    try:
        # Insert the content and its (optional) embedding into the database
        db_response = (
            supabase_admin.table("knowledge_base_entries")
            .insert(data_to_insert)
            .execute() # Removed .select() here as it was problematic before.
        )
        
        if db_response.data: # If insert was successful and returned data (default for RLS bypass with service_role)
            print(f"KB_INGEST: Successfully inserted KB content for org {org_id}. ID: {db_response.data[0].get('id')}")
            # Ensure all fields required by KBEntry are present or optional in the model
            # Manually construct if 'embedding' is not always returned or if types mismatch
            response_data = db_response.data[0]
            return KBEntry( # Use Pydantic model for response validation
                id=UUID(response_data.get("id")), # Cast to UUID
                content=response_data.get("content"),
                organization_id=UUID(response_data.get("organization_id")) if response_data.get("organization_id") else None,
                embedding=response_data.get("embedding"), # Will be None if not generated
                created_at=datetime.fromisoformat(response_data.get("created_at")) if response_data.get("created_at") else None
            )
        else:
            # This case (no data, no error) might indicate an issue or specific PostgREST config
            error_message = "KB insert returned no data, but no error was raised by PostgREST."
            if hasattr(db_response, 'error') and db_response.error: # Should be caught by APIError block though
                error_message = db_response.error.message
            print(f"KB_INGEST: KB insert failed for org {org_id}. Error: {error_message}")
            raise HTTPException(status_code=500, detail=f"Failed to ingest content: {error_message}")

    except APIError as e: # Catch PostgREST errors
        print(f"KB_INGEST: APIError inserting KB content for org {org_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Database error during ingestion: {e.message}")
    except Exception as e: # Catch other unexpected errors
        print(f"KB_INGEST: Unexpected error inserting KB content for org {org_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error during ingestion: {str(e)}")


@app.get("/api/knowledge-base", response_model=List[KBEntry])
async def get_knowledge_base_entries(
    current_user: AuthenticatedUser = Depends(get_current_user_with_org_dependency)
):
    org_id = current_user.organization_id # Ensured by dependency
    try:
        response = (
            supabase_admin.table("knowledge_base_entries")
            .select("id, content, organization_id, created_at") # Not selecting embedding for list view
            .eq("organization_id", str(org_id))
            .order("created_at", desc=True)
            .execute()
        )
        # Convert to KBEntry Pydantic models for type safety and correct formatting
        return [KBEntry(**item) for item in response.data] if response.data else []
    except APIError as e:
        print(f"APIError fetching KB for org {org_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Could not fetch knowledge base: {e.message}")
    except Exception as e:
        print(f"Unexpected error fetching KB for org {org_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error fetching knowledge base: {str(e)}")


@app.delete("/api/knowledge-base/{entry_id}")
async def delete_knowledge_base_entry(
    entry_id: UUID, # Changed to UUID for path param validation
    current_user: AuthenticatedUser = Depends(get_current_user_with_org_dependency)
):
    org_id = current_user.organization_id # Ensured by dependency
    try:
        # First, verify the entry belongs to the user's organization before deleting
        verify_response = (
            supabase_admin.table("knowledge_base_entries")
            .select("id")
            .eq("id", str(entry_id))
            .eq("organization_id", str(org_id))
            .maybe_single() # Ensure it's one or none
            .execute()
        )
        if not verify_response.data:
            raise HTTPException(status_code=404, detail="Entry not found or permission denied.")

        delete_response = (
            supabase_admin.table("knowledge_base_entries")
            .delete()
            .eq("id", str(entry_id)) # Ensure this is the UUID string
            .execute()
        )
        
        # According to PostgREST, a successful DELETE with return=representation (default)
        # returns the deleted row(s). If return=minimal, it's an empty list.
        if delete_response.data: # Check if data (deleted row) is returned
            return {"ok": True, "message": "Entry deleted successfully"}
        # If no data and no error, it means delete was successful with return=minimal or RLS masked it
        # (though service_role bypasses RLS for direct operations like this)
        elif not (hasattr(delete_response, 'error') and delete_response.error):
            return {"ok": True, "message": "Entry deleted (or no data returned post-delete)"}
        else: # Should be caught by APIError if PostgREST itself errored
            error_message = "Delete operation failed or returned unexpected result."
            if hasattr(delete_response, 'error') and delete_response.error:
                 error_message = delete_response.error.message
            raise HTTPException(status_code=500, detail=f"Failed to delete entry: {error_message}")

    except APIError as e: # Catch PostgREST errors
        print(f"APIError deleting KB entry {entry_id} for org {org_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Database error deleting entry: {e.message}")
    except Exception as e: # Catch other errors
        print(f"Unexpected error deleting KB entry {entry_id} for org {org_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error deleting entry: {str(e)}")


#--- AI Response Generation Endpoint ---
@app.post("/api/generate-response", response_model=RAGResponse)
async def generate_ai_response_rag(
    request_data: QueryRequest,
    current_user: AuthenticatedUser = Depends(get_current_user_with_org_dependency)
):
    org_id = current_user.organization_id # Ensured by dependency
    customer_query = request_data.text
    retrieved_context_count = 0
    response_source = "tfidf_failed" # Default pessimistic source

    # 1. Check if knowledge base is empty for this organization
    kb_entry_count = 0
    try:
        kb_check_response = (
            supabase_admin.table("knowledge_base_entries")
            .select("id", count="exact") # Request only count for efficiency
            .eq("organization_id", str(org_id))
            .limit(1) # We only need to know if at least one exists
            .execute()
        )
        # The count is available in kb_check_response.count if using supabase-py v1.x with PostgREST count header
        kb_entry_count = kb_check_response.count if kb_check_response.count is not None else 0
        print(f"KB check for org {org_id}: Found {kb_entry_count} entries.")

        if kb_entry_count == 0:
            return RAGResponse(
                generated_response="The knowledge base for this organization is currently empty. Please ingest content to enable AI responses.",
                retrieved_context_count=0,
                source="no_kb_content"
            )
    except APIError as e: # Catch PostgREST errors
        print(f"APIError checking knowledge base for org {org_id}: {e}")
        # If KB check fails, we might still proceed to TF-IDF which will also fail but more gracefully
    except Exception as e: # Catch other unexpected errors
        print(f"Unexpected error checking knowledge base for org {org_id}: {e}")
        # Proceed cautiously, TF-IDF/RAG will likely fail or use default response

    # 2. Attempt to get OpenAI API key and use OpenAI RAG
    decrypted_api_key = await get_decrypted_openai_api_key(org_id, supabase_admin)

    if decrypted_api_key:
        try:
            print(f"OPENAI_RAG: Attempting OpenAI RAG for org {org_id}")
            openai_client = openai.OpenAI(api_key=decrypted_api_key)
            
            # Generate embedding for the customer query
            embedding_response = openai_client.embeddings.create(
                input=customer_query,
                model="text-embedding-ada-002" # OpenAI's recommended embedding model
            )
            query_embedding = embedding_response.data[0].embedding

            # Call the Supabase Edge Function/RPC for matching documents
            match_response = supabase_admin.rpc(
                "match_documents", # Ensure this function name matches what you created in Supabase
                {
                    "p_query_embedding": query_embedding,
                    "p_match_threshold": 0.7, # Adjust as needed for relevance
                    "p_match_count": 3,       # Number of context chunks to retrieve
                    "p_org_id": str(org_id)
                }
            ).execute()

            if match_response.data: # Check if RPC returned any matching documents
                retrieved_context_count = len(match_response.data)
                print(f"OPENAI_RAG: Found {retrieved_context_count} relevant documents for org {org_id}")
                context_for_prompt = "\n\n---\n\n".join([doc["content"] for doc in match_response.data])
                
                # Construct a prompt for the language model
                prompt = f"""You are a helpful AI customer support agent for an organization. 
Based on the following information from the organization's knowledge base, answer the customer's query. 
If the information isn't sufficient, politely state that you couldn't find a specific answer.

Knowledge Base Context:
---
{context_for_prompt}
---

Customer Query: "{customer_query}"

Answer:"""
                
                # Call OpenAI's chat completion API
                chat_completion_response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo", # Or your preferred model like gpt-4
                    messages=[
                        {"role": "system", "content": "You are a helpful customer support assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3, # Lower temperature for more factual, less creative responses
                )
                generated_text = chat_completion_response.choices[0].message.content.strip()
                response_source = "openai_rag"
                return RAGResponse(generated_response=generated_text, retrieved_context_count=retrieved_context_count, source=response_source)
            else:
                # No relevant documents found by vector search
                print(f"OPENAI_RAG: No relevant documents found for query from org {org_id}")
                response_source = "openai_rag_no_context" # Will fall through to TF-IDF or generic
        
        except openai.APIError as e: # Catch OpenAI specific API errors (quota, auth, etc.)
            print(f"OpenAI API error during RAG generation for org {org_id} (falling back to TF-IDF if possible): {e}")
            # Log the error but allow fallback to TF-IDF
        except Exception as e: # Catch other unexpected errors during OpenAI RAG
            print(f"Unexpected error during OpenAI RAG for org {org_id} (falling back to TF-IDF if possible): {e}")
            # Log error, allow TF-IDF fallback
    else:
        print(f"No OpenAI API key for org {org_id}. Attempting TF-IDF fallback.")

    # 3. TF-IDF Fallback (if OpenAI RAG failed or no key, and KB is not empty)
    # This part will only run if kb_entry_count > 0 AND (decrypted_api_key is None OR OpenAI RAG failed/found no context)
    if kb_entry_count > 0 : # Ensure we only run TF-IDF if there's content
        try:
            print(f"TFIDF_FALLBACK: Attempting TF-IDF for org {org_id}")
            # Fetch all documents for TF-IDF vectorization
            kb_entries_response = (
                supabase_admin.table("knowledge_base_entries")
                .select("id, content") # Only need content for TF-IDF
                .eq("organization_id", str(org_id))
                .execute()
            )

            if kb_entries_response.data and len(kb_entries_response.data) > 0:
                documents = [entry["content"] for entry in kb_entries_response.data]
                
                if not documents: # Should not happen if kb_entries_response.data is checked
                    raise ValueError("No documents found for TF-IDF (should have been caught by kb_entry_count).")

                # Create TF-IDF matrix
                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(documents)
                
                # Transform the customer query
                query_tfidf_vector = vectorizer.transform([customer_query])
                
                # Calculate cosine similarities
                cosine_similarities = cosine_similarity(query_tfidf_vector, tfidf_matrix).flatten()
                
                num_matches_to_retrieve = min(3, len(documents)) # Get top 3 or fewer
                
                # Get indices of top N matches above a certain threshold
                # Filter out matches below the threshold first
                potential_matches_indices = np.where(cosine_similarities > 0.1)[0] # Example threshold, adjust as needed
                
                if len(potential_matches_indices) > 0:
                    # Sort these potential matches by similarity score
                    sorted_potential_matches = sorted(potential_matches_indices, key=lambda i: cosine_similarities[i], reverse=True)
                    top_match_indices = sorted_potential_matches[:num_matches_to_retrieve]
                    
                    relevant_snippets = [documents[i] for i in top_match_indices]
                    retrieved_context_count = len(relevant_snippets)
                    
                    if relevant_snippets:
                        combined_snippets = "\n\n---\n\n".join(relevant_snippets) # Join snippets with a separator
                        fallback_response_text = (
                            f"Based on the knowledge base, here's some information that might be relevant:\n\n"
                            f"{combined_snippets}"
                        )
                        response_source = "tfidf_retrieval"
                        return RAGResponse(generated_response=fallback_response_text, retrieved_context_count=retrieved_context_count, source=response_source)
                    else: # Threshold was too high, or no good matches after sorting (unlikely if potential_matches_indices > 0)
                        response_source = "tfidf_retrieval_no_match"
                else: # No matches above the threshold
                    response_source = "tfidf_retrieval_no_match"
            else: # No documents fetched for TF-IDF (should be caught by kb_entry_count)
                response_source = "tfidf_retrieval_failed" # Indicate failure to retrieve docs for TF-IDF
                print(f"TF-IDF: No knowledge base content found for org {org_id} for TF-IDF processing.")

        except Exception as e:
            print(f"Error during TF-IDF fallback for org {org_id}: {e}")
            response_source = "tfidf_failed" # Mark as failed if TF-IDF itself errors

    # 4. Generic Fallback if all else fails (including if KB was empty and initial check somehow bypassed)
    # Determine the most accurate final source based on what happened
    final_source = response_source # Start with the last known source
    
    # Re-check kb_entry_count for the most accurate final message if all methods failed
    # This simple re-check assumes the initial kb_entry_count is still valid.
    # A more robust way might involve passing a flag from the initial check.
    if kb_entry_count == 0: # This implies the initial check passed but somehow we are here, or it was an error
        final_generated_response = "The knowledge base for this organization is currently empty. Please ingest content to enable AI responses."
        final_source = "no_kb_content"
    elif response_source in ["openai_rag_no_context", "tfidf_retrieval_no_match"]:
        final_generated_response = "I found some information in the knowledge base, but it might not be specific enough to answer your query. A team member will review it."
    elif response_source in ["tfidf_failed", "tfidf_retrieval_failed"]: # TF-IDF errored or found no docs
        final_generated_response = "I'm having a little trouble searching the knowledge base right now. A team member will review your query."
    else: # Default generic if no other condition met (should ideally be one of the above)
        final_generated_response = "I'm sorry, I couldn't find a specific answer at the moment. A team member will review your query."

    return RAGResponse(generated_response=final_generated_response, retrieved_context_count=0, source=final_source)

#--- Email Webhook Endpoint ---
@app.post("/api/email/inbound-webhook")
async def email_inbound_webhook(request: Request): # Use raw request to get JSON
    print("EMAIL_WEBHOOK: Received email webhook request.")
    raw_payload = {}
    try:
        raw_payload = await request.json()
        # For debugging, pretty print the raw payload
        print(f"EMAIL_WEBHOOK: Raw payload: {json.dumps(raw_payload, indent=2)}")
    except json.JSONDecodeError:
        print("EMAIL_WEBHOOK: Failed to decode JSON payload.")
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")

    # Try to parse with Pydantic model (for validation and easier access)
    parsed_email_payload: Optional[InboundEmailPayload] = None
    try:
        # If your actual payload has keys like "Message-ID", Pydantic with alias will handle it.
        parsed_email_payload = InboundEmailPayload(**raw_payload)
        print(f"EMAIL_WEBHOOK: Parsed Pydantic (top-level): To: {parsed_email_payload.to_email}, From: {parsed_email_payload.from_email}, Subject: {parsed_email_payload.subject}")
    except Exception as pydantic_error: # Catch Pydantic validation errors
        print(f"EMAIL_WEBHOOK: Pydantic validation error for webhook payload: {pydantic_error}")
        # Still proceed with raw_payload if Pydantic fails, but log it.
        # Depending on strategy, you might want to raise HTTPException here if strict parsing is required.
        # For now, we'll try to work with raw_payload for routing address.


    #--- Robust Organization ID Extraction ---
    # The actual recipient that your unique email address is (e.g., org_uuid@yourdomain.com)
    # is often in SendGrid's 'envelope' field, specifically 'envelope.to[0]'.
    # Or, it might be in a custom header if you configure SendGrid that way.
    routing_address = None
    if isinstance(raw_payload.get("envelope"), dict): # Check if 'envelope' key exists and is a dict
        envelope_to_list = raw_payload["envelope"].get("to")
        if isinstance(envelope_to_list, list) and len(envelope_to_list) > 0:
            routing_address = envelope_to_list[0] # Get the first email in the 'to' array of envelope
            print(f"EMAIL_WEBHOOK: Routing address from envelope.to[0]: {routing_address}")
    
    if not routing_address and parsed_email_payload: # Fallback to main 'to' field if envelope not found/structured as expected
        routing_address = parsed_email_payload.to_email # This comes from the Pydantic model (aliased from "to")
        print(f"EMAIL_WEBHOOK: Warning: Routing address taken from main 'to' field: {routing_address}. Verify SendGrid payload structure.")
    elif not routing_address and "to" in raw_payload: # Further fallback if Pydantic parsing failed
        # Ensure 'to' is treated as a string if it's directly from raw_payload
        to_field_raw = raw_payload.get("to")
        if isinstance(to_field_raw, list) and len(to_field_raw) > 0: # Some providers send 'to' as a list
            routing_address = str(to_field_raw[0])
        elif isinstance(to_field_raw, str):
            routing_address = to_field_raw
        print(f"EMAIL_WEBHOOK: Warning: Routing address taken from raw 'to' field: {routing_address}. Pydantic parsing may have failed earlier.")


    if not routing_address: # If still no routing_address after all attempts
        print("EMAIL_WEBHOOK: Critical: Could not determine the routing address for organization lookup.")
        raise HTTPException(status_code=400, detail="Could not determine routing address from email payload.")

    extracted_org_identifier_str = None
    try:
        if "@" in routing_address:
            local_part = routing_address.split("@")[0]
            # Example: org_uuid@domain.com -> local_part = org_uuid
            # Or just uuid@domain.com -> local_part = uuid
            if local_part.startswith("org_"):
                extracted_org_identifier_str = local_part[len("org_"):] # Remove "org_" prefix
                print(f"EMAIL_WEBHOOK: Stripped 'org_' prefix, potential UUID: {extracted_org_identifier_str}")
            else:
                extracted_org_identifier_str = local_part # Assume local_part is the UUID

            UUID(extracted_org_identifier_str) # Validate if it's a UUID string after stripping potential prefix
            print(f"EMAIL_WEBHOOK: Extracted organization identifier (string): {extracted_org_identifier_str}")
        else:
            # This case should ideally not happen if it's a valid email address
            print(f"EMAIL_WEBHOOK: Routing address '{routing_address}' does not contain '@'.")
            raise ValueError("Invalid email format for routing address") # Raise ValueError to be caught below

    except ValueError: # Catches UUID validation error or the one raised above
        print(f"EMAIL_WEBHOOK: Could not parse '{routing_address}' to extract a valid organization UUID.")
        raise HTTPException(status_code=400, detail="Could not identify organization from recipient email format.")
    except Exception as e: # Catch any other unexpected error during extraction
        print(f"EMAIL_WEBHOOK: Error extracting org identifier: {e}")
        raise HTTPException(status_code=500, detail="Internal error processing recipient email.")

    if not extracted_org_identifier_str: # Should be caught by exceptions above, but as a safeguard
        print("EMAIL_WEBHOOK: Failed to extract organization identifier string from recipient email.")
        raise HTTPException(status_code=400, detail="Organization identifier string not found in recipient email.")

    # --- Fetch organization_id from DB based on extracted_org_identifier_str ---
    # Here, we assume extracted_org_identifier_str is the actual organization_id (UUID string)
    organization_id_to_store: Optional[UUID] = None
    try:
        # Verify this organization actually exists
        org_lookup_response = supabase_admin.table("organizations").select("id").eq("id", extracted_org_identifier_str).maybe_single().execute()
        if not org_lookup_response.data:
            print(f"EMAIL_WEBHOOK: Organization with ID '{extracted_org_identifier_str}' not found in database.")
            raise HTTPException(status_code=404, detail=f"Organization not found for identifier: {extracted_org_identifier_str}")
        organization_id_to_store = UUID(org_lookup_response.data["id"])
    except APIError as e:
        print(f"EMAIL_WEBHOOK: Database error looking up organization {extracted_org_identifier_str}: {e}")
        raise HTTPException(status_code=500, detail="Database error during organization lookup.")
    
    if not organization_id_to_store: # Should be caught by exceptions, but safeguard
        print(f"EMAIL_WEBHOOK: Organization ID could not be confirmed for identifier {extracted_org_identifier_str}")
        raise HTTPException(status_code=404, detail="Organization ID confirmation failed.")

    # Extract sender name and actual email
    sender_name_extracted = None
    # Use from_email from Pydantic model if parsing was successful, else from raw_payload
    from_field_to_parse = parsed_email_payload.from_email if parsed_email_payload else str(raw_payload.get("from", ""))
    actual_sender_email = from_field_to_parse # Default to the full string

    if "<" in from_field_to_parse and ">" in from_field_to_parse:
        try:
            parts = from_field_to_parse.rsplit("<", 1)
            sender_name_extracted = parts[0].strip().replace('"', '') # Remove quotes if any
            actual_sender_email = parts[1].rstrip(">").strip()
        except Exception as e: # Catch potential errors if format is unexpected
            print(f"EMAIL_WEBHOOK: Could not parse sender name/email from '{from_field_to_parse}': {e}")
    
    # Extract Message-ID and In-Reply-To for threading
    # These are often in a 'headers' sub-dictionary or string, depending on the webhook provider
    incoming_message_id: Optional[str] = None
    incoming_in_reply_to: Optional[str] = None
    
    # Try to get headers from common SendGrid structure or Pydantic model
    raw_headers_field = raw_payload.get("headers") # SendGrid often provides 'headers' as a flat string
    email_headers = {} # To store parsed headers

    if isinstance(raw_headers_field, dict): # If headers is already a dictionary
        email_headers = {k.lower(): v for k,v in raw_headers_field.items()} # Normalize keys
    elif isinstance(raw_headers_field, str): # If it's a flat string of 'key: value' lines
        print("EMAIL_WEBHOOK: 'headers' field is a string, attempting basic parsing.")
        try:
            for line in raw_headers_field.splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    email_headers[key.strip().lower()] = value.strip() # Normalize key
        except Exception as e_header_parse:
            print(f"EMAIL_WEBHOOK: Error parsing flat 'headers' string: {e_header_parse}")

    # Get standard headers, preferring parsed email_headers if available
    incoming_message_id = email_headers.get("message-id")
    incoming_in_reply_to = email_headers.get("in-reply-to")
    
    # Fallback to Pydantic model fields if headers dict/string didn't yield them
    if not incoming_message_id and parsed_email_payload:
        incoming_message_id = parsed_email_payload.message_id
    if not incoming_in_reply_to and parsed_email_payload:
        incoming_in_reply_to = parsed_email_payload.in_reply_to

    print(f"EMAIL_WEBHOOK: Final Incoming Message-ID: {incoming_message_id}, In-Reply-To: {incoming_in_reply_to}")

    # Check for duplicates using Message-ID
    if incoming_message_id:
        try:
            duplicate_check = supabase_admin.table("query_messages").select("id").eq("message_id_header", incoming_message_id).limit(1).execute()
            if duplicate_check.data:
                print(f"EMAIL_WEBHOOK: Duplicate email detected by Message-ID {incoming_message_id}. Skipping.")
                return {"message": "Duplicate email. Already processed."}
        except APIError as e:
            print(f"EMAIL_WEBHOOK: Error checking for duplicate Message-ID: {e}")
            # Decide if to proceed or fail; for now, proceed but log

    existing_query_id: Optional[UUID] = None
    if incoming_in_reply_to: # If it's potentially a reply
        try:
            # Find a message in query_messages that has this In-Reply-To value as its own Message-ID
            parent_message_response = (
                supabase_admin.table("query_messages")
                .select("customer_query_id") # We need the ID of the parent query
                .eq("message_id_header", incoming_in_reply_to) # Match the In-Reply-To with a stored Message-ID
                .eq("organization_id", str(organization_id_to_store)) # Ensure it's within the same org
                .maybe_single() # There should only be one such original message
                .execute()
            )
            if parent_message_response.data:
                existing_query_id = UUID(parent_message_response.data["customer_query_id"])
                print(f"EMAIL_WEBHOOK: Found existing query thread {existing_query_id} based on In-Reply-To.")
        except APIError as e: # Catch PostgREST errors
            print(f"EMAIL_WEBHOOK: APIError looking up parent message by In-Reply-To: {e}")
        except Exception as e: # Catch other errors like UUID conversion
            print(f"EMAIL_WEBHOOK: Error processing In-Reply-To: {e}")
    
    # Use a consistent subject and body, defaulting if None from Pydantic parsed object or raw
    email_subject = (parsed_email_payload.subject if parsed_email_payload and parsed_email_payload.subject 
                     else raw_payload.get("subject", "No Subject"))
    email_text_body = (parsed_email_payload.text if parsed_email_payload and parsed_email_payload.text 
                       else raw_payload.get("text", "No body content."))


    try:
        if existing_query_id:
            # Add new message to the existing query thread
            print(f"EMAIL_WEBHOOK: Adding message to existing query {existing_query_id}")
            message_payload = {
                "customer_query_id": str(existing_query_id),
                "sender_type": "customer", # Assuming incoming email is from customer
                "sender_identifier": actual_sender_email,
                "body_text": email_text_body,
                "message_id_header": incoming_message_id, # Store the Message-ID of this new email
                "in_reply_to_header": incoming_in_reply_to, # Store the In-Reply-To of this new email
                "organization_id": str(organization_id_to_store) # Add org_id to query_messages
            }
            insert_message_response = supabase_admin.table("query_messages").insert(message_payload).execute() # Removed .select()
            if not insert_message_response.data: # If insert failed or returned no data (unexpected for default RLS with service_role)
                raise APIError({"message":"Failed to insert message into existing thread or no data returned."}) # Will be caught by APIError handler

            # Update parent query's timestamp and status
            (supabase_admin.table("customer_queries")
                .update({
                    "status": "customer_reply", # Mark that customer has replied
                    "updated_at": datetime.utcnow().isoformat()
                })
                .eq("id", str(existing_query_id))
                .execute()) # Fire and forget update, or check response
            print(f"EMAIL_WEBHOOK: Successfully added reply to query {existing_query_id}")
            return {"message": "Email reply processed and added to existing query", "query_id": str(existing_query_id), "message_id": insert_message_response.data[0]['id']}
        else:
            # Create new query and its first message
            print(f"EMAIL_WEBHOOK: Creating new query for organization {organization_id_to_store}")
            customer_query_payload = {
                "organization_id": str(organization_id_to_store),
                "channel": "email",
                "sender_identifier": actual_sender_email,
                "sender_name": sender_name_extracted,
                "subject": email_subject,
                "body_text": email_text_body, # Store the first message's body here
                "status": "new",
                "updated_at": datetime.utcnow().isoformat() # Set initial updated_at
                # received_at defaults to now() in DB schema
            }
            insert_query_response = supabase_admin.table("customer_queries").insert(customer_query_payload).execute() # Removed .select()
            if not insert_query_response.data:
                raise APIError({"message":"Failed to create new customer query or no data returned."})
            
            new_query_id = UUID(insert_query_response.data[0]["id"])
            print(f"EMAIL_WEBHOOK: New query created with ID: {new_query_id}")

            # Create the first message in query_messages
            message_payload = {
                "customer_query_id": str(new_query_id),
                "sender_type": "customer",
                "sender_identifier": actual_sender_email,
                "body_text": email_text_body,
                "message_id_header": incoming_message_id, # Store Message-ID of this first email
                "in_reply_to_header": incoming_in_reply_to, # Might be None for first email
                "organization_id": str(organization_id_to_store) # Add org_id
            }
            insert_message_response = supabase_admin.table("query_messages").insert(message_payload).execute() # Removed .select()
            if not insert_message_response.data:
                # If this fails, the parent query was created but its first message wasn't.
                # This could lead to an orphaned query. Consider rollback or cleanup logic.
                raise APIError({"message":"Failed to insert initial message for new query or no data returned."})

            print(f"EMAIL_WEBHOOK: Successfully inserted initial message for new query {new_query_id}")
            return {"message": "Email processed and new query created successfully", "query_id": str(new_query_id), "message_id": insert_message_response.data[0]['id']}

    except APIError as e: # Catch PostgREST errors from Supabase calls
        print(f"EMAIL_WEBHOOK: APIError during query/message processing: {e.json() if hasattr(e, 'json') else str(e)}")
        # Try to use PostgREST error message if available
        error_detail = e.message if hasattr(e, 'message') else str(e)
        raise HTTPException(status_code=500, detail=f"Database error processing email thread: {error_detail}")
    except Exception as e: # Catch other unexpected errors
        print(f"EMAIL_WEBHOOK: Unexpected error during query/message processing: {type(e).__name__} - {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error processing email thread: {str(e)}")

#--- Endpoint to fetch customer queries for the frontend ---
@app.get("/api/customer-queries", response_model=List[CustomerQueryDB])
async def get_customer_queries(
    current_user: AuthenticatedUser = Depends(get_current_user_with_org_dependency)
):
    org_id = current_user.organization_id # Ensured by dependency
    try:
        response = (
            supabase_admin.table("customer_queries")
            .select("*") # Select all fields for the query list
            .eq("organization_id", str(org_id))
            .order("updated_at", desc=True) # Show most recently updated queries first
            .limit(100) # Basic pagination for now
            .execute()
        )
        # Convert to Pydantic models for type safety and correct formatting
        return [CustomerQueryDB(**item) for item in response.data] if response.data else []
    except APIError as e:
        print(f"APIError fetching customer queries for org {org_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Could not fetch customer queries: {e.message}")
    except Exception as e:
        print(f"Unexpected error fetching customer queries for org {org_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error fetching customer queries: {str(e)}")

#--- New Endpoint to fetch messages for a specific query ---
@app.get("/api/customer-queries/{query_id}/messages", response_model=List[QueryMessageDB])
async def get_query_messages(
    query_id: UUID, # Path parameter for the query ID
    current_user: AuthenticatedUser = Depends(get_current_user_with_org_dependency)
):
    org_id = current_user.organization_id # Ensured by dependency
    # First, verify the query belongs to the user's organization for security
    try:
        query_check_response = supabase_admin.table("customer_queries").select("id").eq("id", str(query_id)).eq("organization_id", str(org_id)).maybe_single().execute()
        if not query_check_response.data:
            raise HTTPException(status_code=404, detail="Parent query not found or not accessible.")
    except APIError as e: # Catch DB errors during the check
        raise HTTPException(status_code=500, detail=f"Error verifying query: {e.message}")

    # If query check passes, fetch the messages
    try:
        messages_response = (
            supabase_admin.table("query_messages")
            .select("*") # Select all fields for messages
            .eq("customer_query_id", str(query_id))
            # .eq("organization_id", str(org_id)) # Already ensured by parent query check, but good for direct security
            .order("created_at", desc=False) # Show oldest messages first for chat flow
            .execute()
        )
        # Convert to Pydantic models
        return [QueryMessageDB(**item) for item in messages_response.data] if messages_response.data else []
    except APIError as e:
        print(f"APIError fetching messages for query {query_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Could not fetch messages: {e.message}")
    except Exception as e:
        print(f"Unexpected error fetching messages for query {query_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error fetching messages: {str(e)}")


#--- Endpoint to Process/Update a Customer Query (e.g. save AI draft, change status) ---
@app.patch("/api/customer-queries/{query_id}/process", response_model=CustomerQueryDB)
async def process_customer_query(
    query_id: UUID,
    process_data: ProcessQueryRequest, # Contains AI draft details and new_status
    current_user: AuthenticatedUser = Depends(get_current_user_with_org_dependency)
):
    org_id = current_user.organization_id # Ensured by dependency
    print(f"Processing query {query_id} for org {org_id} with status {process_data.new_status}")

    # Verify query belongs to the user's organization
    try:
        verify_response = (
            supabase_admin.table("customer_queries")
            .select("id")
            .eq("id", str(query_id))
            .eq("organization_id", str(org_id))
            .maybe_single()
            .execute()
        )
        if not verify_response.data:
            raise HTTPException(status_code=404, detail="Query not found or permission denied.")
    except APIError as e:
        print(f"APIError verifying query {query_id} for org {org_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Database error verifying query: {e.message}")

    # Prepare payload for updating the customer_queries table
    update_payload = {
        "ai_draft_response": process_data.ai_draft_response,
        "ai_response_source": process_data.ai_response_source,
        "ai_retrieved_context_count": process_data.ai_retrieved_context_count,
        "status": process_data.new_status,
        "updated_at": datetime.utcnow().isoformat() # Always update this on process
    }
    try:
        update_response = (
            supabase_admin.table("customer_queries")
            .update(update_payload)
            .eq("id", str(query_id))
            .execute() # Removed .select()
        )
        
        # After update, PostgREST with default Prefer: return=representation should return updated row(s)
        if update_response.data: 
            print(f"Successfully processed query {query_id}")
            return CustomerQueryDB(**update_response.data[0]) # Validate with Pydantic
        else:
            # This path might be hit if PostgREST is configured with Prefer: return=minimal
            # or if RLS with service_role has some unexpected interaction.
            # For robustness, re-fetch the record to ensure we return the updated state.
            print(f"Warning: Update for query {query_id} returned no data directly. Re-fetching.")
            refetched_query_response = supabase_admin.table("customer_queries").select("*").eq("id", str(query_id)).single().execute() # Use single to expect one
            if refetched_query_response.data:
                print(f"Successfully processed query {query_id}, re-fetched for response.")
                return CustomerQueryDB(**refetched_query_response.data)
            
            # If re-fetch also fails or returns no data (should not happen if update was silent success)
            error_message = "Query update returned no data, and re-fetch failed."
            if hasattr(update_response, 'error') and update_response.error: # Check original response for error
                error_message = update_response.error.message
            print(f"Failed to process query {query_id}. Error: {error_message}")
            raise HTTPException(status_code=500, detail=f"Failed to process query: {error_message}")

    except APIError as e: # Catch PostgREST errors
        print(f"APIError processing query {query_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Database error processing query: {e.message}")
    except Exception as e: # Catch other errors
        print(f"Unexpected error processing query {query_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error processing query: {str(e)}")

#--- New Endpoint to Save Agent Reply ---
@app.post("/api/customer-queries/{query_id}/agent-reply", response_model=QueryMessageDB)
async def save_agent_reply(
    query_id: UUID,
    reply_data: AgentReplyRequest,
    current_user: AuthenticatedUser = Depends(get_current_user_with_org_dependency)
):
    org_id = current_user.organization_id # Ensured by dependency
    agent_user_id = current_user.id # The logged-in Supabase user ID

    print(f"AGENT_REPLY: Agent {agent_user_id} replying to query {query_id} for org {org_id}")

    # 1. Verify the parent query exists and belongs to the organization
    parent_query_dict: Optional[Dict[str, Any]] = None # To store fetched parent query
    try:
        query_check_response = (
            supabase_admin.table("customer_queries")
            .select("id, status, subject, sender_identifier") # Fetch fields needed for email logic
            .eq("id", str(query_id))
            .eq("organization_id", str(org_id))
            .maybe_single() # Expect one or none
            .execute()
        )
        if not query_check_response.data: # If no query found
            raise HTTPException(status_code=404, detail="Parent query not found or not accessible for this organization.")
        parent_query_dict = query_check_response.data # Store the fetched query data
    except APIError as e:
        print(f"AGENT_REPLY: APIError verifying parent query {query_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Database error verifying query: {e.message}")

    # 2. Prepare and insert the agent's message
    message_payload = {
        "customer_query_id": str(query_id),
        "organization_id": str(org_id), # Also store org_id in messages for easier scoping
        "sender_type": "agent",
        "sender_identifier": str(agent_user_id), # Agent's user ID
        "body_text": reply_data.reply_text,
        # message_id_header and in_reply_to_header are typically for incoming customer emails, not agent replies
    }
    try:
        insert_message_response = (
            supabase_admin.table("query_messages")
            .insert(message_payload)
            .execute() # Removed .select()
        )
        # Check if insert was successful and returned data (default for RLS bypass with service_role)
        if not insert_message_response.data: 
            error_message = "Agent reply insert returned no data."
            if hasattr(insert_message_response, 'error') and insert_message_response.error:
                error_message = insert_message_response.error.message
            print(f"AGENT_REPLY: Failed to insert agent reply for query {query_id}. Error: {error_message}")
            raise HTTPException(status_code=500, detail=f"Failed to save agent reply: {error_message}")
        
        created_message_data = insert_message_response.data[0]
        print(f"AGENT_REPLY: Agent reply saved with ID: {created_message_data['id']}")

        # 3. Update the parent customer_query status and updated_at
        new_query_status = "agent_replied" # Or "closed" if this reply resolves it
        update_query_payload = {
            "status": new_query_status,
            "updated_at": datetime.utcnow().isoformat(), # Update timestamp
            "ai_draft_response": None, # Clear AI draft as agent has replied
            "ai_response_source": None,
            "ai_retrieved_context_count": None
        }
        ( # Fire and forget update, or add error checking if needed
            supabase_admin.table("customer_queries")
            .update(update_query_payload)
            .eq("id", str(query_id))
            .execute()
        )
        print(f"AGENT_REPLY: Parent query {query_id} status updated to {new_query_status}")

        # 4. Send the actual email to the customer (using parent_query_dict)
        customer_email_address = parent_query_dict.get("sender_identifier") if parent_query_dict else None
        
        # Attempt to parse out just the email from "Name <email@example.com>" format
        if customer_email_address and "<" in customer_email_address and ">" in customer_email_address:
            try:
                actual_email = customer_email_address.split("<")[1].split(">")[0]
                customer_email_address = actual_email
            except IndexError:
                print(f"AGENT_REPLY: Could not parse email from sender_identifier: {customer_email_address}")
                # Fallback to using the full string, or handle error more gracefully by not sending

        original_subject = parent_query_dict.get("subject", "your query") if parent_query_dict else "your query"

        # Get the Message-ID of the last customer message in this thread for In-Reply-To
        last_customer_message_id: Optional[str] = None
        try:
            last_cust_msg_resp = supabase_admin.table("query_messages") \
                .select("message_id_header") \
                .eq("customer_query_id", str(query_id)) \
                .eq("sender_type", "customer") \
                .order("created_at", desc=True) \
                .limit(1) \
                .maybe_single() \
                .execute()
            if last_cust_msg_resp.data and last_cust_msg_resp.data.get("message_id_header"):
                last_customer_message_id = last_cust_msg_resp.data["message_id_header"]
        except Exception as e_msg_id:
            print(f"AGENT_REPLY: Error fetching last customer message_id: {e_msg_id}")


        if customer_email_address and SENDER_EMAIL_ADDRESS and SENDGRID_API_KEY:
            email_sent_successfully = await send_actual_email(
                to_email=customer_email_address,
                subject=f"Re: {original_subject}",
                plain_text_content=reply_data.reply_text,
                html_content=f"<p>{reply_data.reply_text.replace(os.linesep, '<br>')}</p>", # Simple HTML version
                in_reply_to_header_val=last_customer_message_id, # For threading
                references_header_val=last_customer_message_id # For threading, can be more complex for deeper threads
            )
            if email_sent_successfully:
                print(f"AGENT_REPLY: Email successfully dispatched to {customer_email_address}")
            else:
                print(f"AGENT_REPLY: Failed to dispatch email to {customer_email_address}. Reply saved to DB only.")
                # You might want to notify the agent in the UI if email sending fails
        else:
            print("AGENT_REPLY: Email sending skipped due to missing customer email or SendGrid config.")
        
        # Convert created_message_data dates for Pydantic if they are strings
        created_message_data['created_at'] = datetime.fromisoformat(created_message_data['created_at'])
        return QueryMessageDB(**created_message_data) # Use Pydantic model

    except APIError as e: # Catch PostgREST errors
        print(f"AGENT_REPLY: APIError saving agent reply for query {query_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Database error saving agent reply: {e.message}")
    except Exception as e: # Catch other errors
        print(f"AGENT_REPLY: Unexpected error saving agent reply for query {query_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error saving agent reply: {str(e)}")


#--- New Dashboard Stats Endpoint ---
@app.get("/api/dashboard/stats", response_model=DashboardStats)
async def get_dashboard_stats(
    current_user: AuthenticatedUser = Depends(get_current_user_with_org_dependency)
):
    org_id = current_user.organization_id # Ensured by dependency
    print(f"DASHBOARD_STATS: Fetching stats for org {org_id}")
    try:
        # Initialize counts
        total_queries = 0
        new_q = 0
        agent_r_q = 0
        customer_r_q = 0
        closed_q = 0 # Assuming 'closed' is a possible status

        # Fetch total queries count
        count_response = (
            supabase_admin.table("customer_queries")
            .select("id", count="exact") # Request only count
            .eq("organization_id", str(org_id))
            .execute()
        )
        total_queries = count_response.count if count_response.count is not None else 0
        
        # Fetch counts for each status individually for simplicity and clarity
        status_list = ["new", "agent_replied", "customer_reply", "closed"] # Define statuses you want to count
        status_counts = {status: 0 for status in status_list}

        for status_to_count in status_list:
            response = (
                supabase_admin.table("customer_queries")
                .select("id", count="exact") # Request only count for efficiency
                .eq("organization_id", str(org_id))
                .eq("status", status_to_count)
                .execute()
            )
            status_counts[status_to_count] = response.count if response.count is not None else 0
        
        new_q = status_counts.get("new", 0)
        agent_r_q = status_counts.get("agent_replied", 0)
        customer_r_q = status_counts.get("customer_reply", 0)
        closed_q = status_counts.get("closed", 0)

        print(f"DASHBOARD_STATS: Counts - Total: {total_queries}, New: {new_q}, AgentReplied: {agent_r_q}, CustomerReply: {customer_r_q}, Closed: {closed_q}")

        return DashboardStats(
            total_queries=total_queries,
            new_queries=new_q,
            agent_replied_queries=agent_r_q,
            customer_reply_queries=customer_r_q,
            closed_queries=closed_q
        )
    except APIError as e:
        print(f"DASHBOARD_STATS: APIError fetching stats for org {org_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Database error fetching stats: {e.message}")
    except Exception as e:
        print(f"DASHBOARD_STATS: Unexpected error fetching stats for org {org_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error fetching stats: {str(e)}")

#--- New Dashboard Query Volume Endpoint ---
@app.get("/api/dashboard/query-volume", response_model=QueryVolumeResponse)
async def get_query_volume(
    days: int = FastAPIQuery(7, ge=1, le=90), # Get 'days' from query param, default 7, min 1, max 90
    current_user: AuthenticatedUser = Depends(get_current_user_with_org_dependency)
):
    org_id = current_user.organization_id # Ensured by dependency
    print(f"DASHBOARD_VOLUME: Fetching query volume for org {org_id} for the last {days} days.")
    
    volume_data: List[QueryVolumeDataPoint] = []
    # Create a list of all dates in the period to ensure all days are represented, even with 0 queries
    date_today = datetime.utcnow().date() # Use UTC date for consistency
    # Generate dates in ascending order for the chart (oldest to newest)
    all_dates_in_period = [(date_today - timedelta(days=i)) for i in range(days - 1, -1, -1)] 

    try:
        # Initialize counts for all dates in the period to 0
        counts_by_date: Dict[str, int] = {dt.isoformat(): 0 for dt in all_dates_in_period}

        # Define the date range for the SQL query (UTC)
        # Start of the first day in the period
        start_date_dt = datetime.combine(all_dates_in_period[0], datetime.min.time(), tzinfo=timezone.utc)
        # End of the last day in the period (exclusive for '<' comparison, so start of next day)
        end_date_dt = datetime.combine(all_dates_in_period[-1] + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
        
        print(f"DASHBOARD_VOLUME: Querying 'received_at' between {start_date_dt.isoformat()} and {end_date_dt.isoformat()}")

        # Fetch queries within the date range
        response = (
            supabase_admin.table("customer_queries")
            .select("received_at") # Only need received_at for grouping
            .eq("organization_id", str(org_id))
            .gte("received_at", start_date_dt.isoformat()) # Greater than or equal to start of the period
            .lt("received_at", end_date_dt.isoformat()) # Less than start of the day AFTER the period
            .execute()
        )

        if response.data:
            for item in response.data:
                received_at_str = item.get("received_at")
                if received_at_str:
                    try:
                        # Ensure timestamp from DB (assumed to be timestamptz, so UTC) is parsed correctly
                        # Convert to date object in UTC
                        if isinstance(received_at_str, str):
                            # Handle potential 'Z' for UTC or offset like +00:00
                            dt_aware = datetime.fromisoformat(received_at_str.replace('Z', '+00:00'))
                            date_obj = dt_aware.astimezone(timezone.utc).date() # Convert to UTC date
                        elif isinstance(received_at_str, datetime): # If already a datetime object
                            date_obj = received_at_str.astimezone(timezone.utc).date() # Ensure UTC date
                        else:
                            raise ValueError("Unsupported received_at format")
                            
                        date_part_iso = date_obj.isoformat() # Get YYYY-MM-DD string (UTC date)
                        if date_part_iso in counts_by_date:
                            counts_by_date[date_part_iso] += 1
                        # else:
                        # This case might happen if DB returns dates outside the pre-calculated range due to timezone nuances
                        # or if `all_dates_in_period` logic has a slight off-by-one.
                        # For simplicity, we only increment if the date matches our expected period.
                        # print(f"DASHBOARD_VOLUME: Date {date_part_iso} from query not in expected range (should not happen often).")
                    except ValueError as ve:
                        print(f"DASHBOARD_VOLUME: Error parsing date '{received_at_str}': {ve}")
        
        # Construct the final list, ensuring all dates in the period are included, ordered chronologically
        for dt_obj in all_dates_in_period: # all_dates_in_period is already sorted from oldest to newest
            date_str = dt_obj.isoformat()
            volume_data.append(QueryVolumeDataPoint(date=date_str, query_count=counts_by_date.get(date_str, 0)))
        
        print(f"DASHBOARD_VOLUME: Processed volume data: {volume_data}")
        return QueryVolumeResponse(data=volume_data, period_days=days)

    except APIError as e:
        print(f"DASHBOARD_VOLUME: APIError fetching query volume for org {org_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Database error fetching query volume: {e.message}")
    except Exception as e:
        print(f"DASHBOARD_VOLUME: Unexpected error fetching query volume for org {org_id}: {type(e).__name__} - {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error fetching query volume: {str(e)}")


#--- Placeholder for running the app with Uvicorn (for local development) ---
# if __name__ == "__main__":
#     import uvicorn
#     # Ensure the host is 0.0.0.0 to be accessible on your network if needed, or 127.0.0.1 for purely local
#     uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
