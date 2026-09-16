"""Amazon Q profile listing — the one call IdC login needs.

An IAM Identity Center sign-in yields an SSO-OIDC access token, but KAS routes
enterprise traffic by ``profile ARN`` (the ``X-Kiro-Profile-Arn`` header), which the
token itself does not carry. The desktop clients resolve it by calling
``ListAvailableProfiles`` with the fresh token; we do the same. Contract (AWS JSON
1.0, bearer auth) read out of the client kiro-cli itself ships
(``amzn_codewhisperer_client``, target prefix ``AmazonCodeWhispererService``):

  POST <see _PROFILE_ENDPOINTS>
    Content-Type: application/x-amz-json-1.0
    X-Amz-Target: AmazonCodeWhispererService.ListAvailableProfiles
    Authorization: Bearer <accessToken>
    body: {"maxResults": ...}
  -> {"profiles": [{"arn": ..., "profileName": ...}, ...], "nextToken": ...}

Verified by probe against the live service: omitting the Authorization header answers
HTTP 400 ``com.amazon.aws.codewhisperer#ValidationException`` "Missing bearer token in
the authorization header", so the host, target and protocol are right and only the
credential is absent. The previous ``KiroControlPlaneBearerService`` target answers
HTTP 400 ``com.amazon.coral.service#UnknownOperationException`` -- the same reply a
nonsense operation name gets, i.e. it is not an operation this service has.

The login's own IdC region is NOT a usable host here: only two commercial endpoints
exist, so an eu-west-1 tenant has no eu-west-1 endpoint to ask. Both are queried and
the ARN a profile carries names its region downstream. Only module constants reach
the hostname, so no caller-supplied region can redirect this request.

An INVALID or expired bearer token does not fail: it answers HTTP 200 with
``{"profiles": []}``, indistinguishable from an account that simply has no Q Developer
profile. Both therefore surface as "no available profiles" rather than as an auth
error. That is honest for the subscription case and merely unhelpful for the bad-token
case, which the login flow should not produce -- the token here was minted seconds
earlier by SSO-OIDC.

Every failure path surfaces as ControlPlaneError rather than a stored-but-unusable
credential -- a connection-level failure included, because an escaping
``aiohttp.ClientError`` is reported by the poll route as "the Kiro auth service is
unreachable", which this service is not.

NOT verified against a live enterprise tenant: no bearer token for an account holding
a Q Developer profile was available, so a successful non-empty profile listing -- and
the ARN's subsequent acceptance by KAS -- remains unproven.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import aiohttp

from kiro_crew.auth.login.endpoints import USER_AGENT

logger = logging.getLogger(__name__)

# One page is plenty: the common enterprise case is a single profile, and callers
# that see several pick the first (multi-profile selection is a documented follow-up).
_MAX_RESULTS = 10
_TARGET = "AmazonCodeWhispererService.ListAvailableProfiles"


class ControlPlaneError(Exception):
    """Profile resolution against the Kiro control plane failed."""


@dataclass
class KiroProfile:
    arn: str
    profile_name: str


# Read out of the client kiro-cli itself ships. The host names are NOT uniform -- the
# us-east-1 endpoint is `codewhisperer.`, not `q.` -- so this cannot be a format string,
# which is what made the previous `q.<region>` pattern wrong for us-east-1. The same
# client carries q.us-gov-{east,west}-1; GovCloud is deliberately absent here because
# the rest of this login path has no GovCloud deployment to route to.
_PROFILE_ENDPOINTS = {
    "us-east-1": "https://codewhisperer.us-east-1.amazonaws.com/",
    "eu-central-1": "https://q.eu-central-1.amazonaws.com/",
}

PROFILE_REGIONS = tuple(_PROFILE_ENDPOINTS)


def control_plane_url(region: str) -> str:
    try:
        return _PROFILE_ENDPOINTS[region]
    except KeyError:
        raise ControlPlaneError(
            f"no ListAvailableProfiles endpoint is published for region {region}"
        ) from None


async def list_available_profiles(
    access_token: str, *, session: aiohttp.ClientSession
) -> list[KiroProfile]:
    """Return the caller's Kiro profiles from the first profile region that has any.

    Raises ControlPlaneError only when NO region could answer: an IdC login without a
    resolvable profile ARN is unusable, so failures must be loud, not stored. A region
    that answers with an empty list is an answer — the caller reports "no available
    profiles" rather than an outage.
    """
    failures: list[str] = []
    for region in PROFILE_REGIONS:
        try:
            profiles = await _list_in_region(access_token, region=region, session=session)
        except ControlPlaneError as err:
            failures.append(f"{region}: {err}")
            continue
        if profiles:
            return profiles
    if len(failures) == len(PROFILE_REGIONS):
        raise ControlPlaneError(
            "ListAvailableProfiles failed in every profile region: " + "; ".join(failures)
        )
    return []


async def _list_in_region(
    access_token: str, *, region: str, session: aiohttp.ClientSession
) -> list[KiroProfile]:
    headers = {
        "Content-Type": "application/x-amz-json-1.0",
        "X-Amz-Target": _TARGET,
        "Authorization": f"Bearer {access_token}",
        "User-Agent": USER_AGENT,
    }
    url = control_plane_url(region)
    try:
        async with session.post(url, json={"maxResults": _MAX_RESULTS}, headers=headers) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise ControlPlaneError(
                    f"ListAvailableProfiles failed: HTTP {resp.status} {body[:500]}"
                )
            try:
                # content_type=None: AWS JSON 1.0 replies carry
                # `application/x-amz-json-1.0`, which aiohttp's default
                # application/json gate would reject as ContentTypeError.
                data = await resp.json(content_type=None)
            except (aiohttp.ClientError, ValueError) as err:
                raise ControlPlaneError(
                    "ListAvailableProfiles returned an undecodable body"
                ) from err
    except (aiohttp.ClientError, asyncio.TimeoutError) as err:
        raise ControlPlaneError(
            f"ListAvailableProfiles could not complete at {url}: {err}"
        ) from err
    if not isinstance(data, dict):
        raise ControlPlaneError("ListAvailableProfiles returned a non-object body")
    raw = data.get("profiles")
    if not isinstance(raw, list):
        raise ControlPlaneError("ListAvailableProfiles response has no 'profiles' list")
    profiles: list[KiroProfile] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        arn = entry.get("arn")
        if isinstance(arn, str) and arn:
            profiles.append(KiroProfile(arn=arn, profile_name=str(entry.get("profileName") or "")))
    return profiles
