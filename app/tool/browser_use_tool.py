import asyncio
import base64
import datetime
import json
import logging
from pathlib import Path
from typing import Generic, Optional, TypeVar

from browser_use import Browser as BrowserUseBrowser
from browser_use import BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from browser_use.dom.service import DomService
from pydantic import Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from app.config import config
from app.llm import LLM
from app.tool.base import BaseTool, ToolResult
from app.tool.web_search import WebSearch

_BROWSER_DESCRIPTION = """\
A powerful browser automation tool that allows interaction with web pages through various actions.
* This tool provides commands for controlling a browser session, navigating web pages, and extracting information
* It maintains state across calls, keeping the browser session alive until explicitly closed
* Use this when you need to browse websites, fill forms, click buttons, extract content, or perform web searches
* Each action requires specific parameters as defined in the tool's dependencies

Key capabilities include:
* Navigation: Go to specific URLs, go back, search the web, or refresh pages
* Interaction: Click elements, input text, select from dropdowns, send keyboard commands
* Scrolling: Scroll up/down by pixel amount or scroll to specific text
* Content extraction: Extract and analyze content from web pages based on specific goals
* Tab management: Switch between tabs, open new tabs, or close tabs

Note: When using element indices, refer to the numbered elements shown in the current browser state.
"""

logger = logging.getLogger(__name__)
Context = TypeVar("Context")


class BrowserUseTool(BaseTool, Generic[Context]):
    name: str = "browser_use"
    description: str = _BROWSER_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "go_to_url",
                    "click_element",
                    "input_text",
                    "scroll_down",
                    "scroll_up",
                    "scroll_to_text",
                    "send_keys",
                    "get_dropdown_options",
                    "select_dropdown_option",
                    "go_back",
                    "web_search",
                    "wait",
                    "extract_content",
                    "switch_tab",
                    "open_tab",
                    "close_tab",
                    "search_video_content",
                ],
                "description": "The browser action to perform. Required parameters by action: go_to_url (url), click_element (index), input_text (index, text), scroll_down/scroll_up (scroll_amount), scroll_to_text (text), send_keys (keys), get_dropdown_options (index), select_dropdown_option (index, text), switch_tab (tab_id), open_tab (url), web_search (query), wait (seconds), extract_content (goal), search_video_content (query). Actions go_back and close_tab require no additional parameters.",
            },
            "url": {
                "type": "string",
                "description": "URL for 'go_to_url' or 'open_tab' actions",
            },
            "index": {
                "type": "integer",
                "description": "Element index for 'click_element', 'input_text', 'get_dropdown_options', or 'select_dropdown_option' actions",
            },
            "text": {
                "type": "string",
                "description": "Text for 'input_text', 'scroll_to_text', or 'select_dropdown_option' actions",
            },
            "scroll_amount": {
                "type": "integer",
                "description": "Pixels to scroll (positive for down, negative for up) for 'scroll_down' or 'scroll_up' actions",
            },
            "tab_id": {
                "type": "integer",
                "description": "Tab ID for 'switch_tab' action",
            },
            "query": {
                "type": "string",
                "description": "Search query for 'web_search' action",
            },
            "goal": {
                "type": "string",
                "description": "Extraction goal for 'extract_content' action",
            },
            "keys": {
                "type": "string",
                "description": "Keys to send for 'send_keys' action",
            },
            "seconds": {
                "type": "integer",
                "description": "Seconds to wait for 'wait' action",
            },
        },
        "required": ["action"],
    }

    lock: asyncio.Lock = Field(default_factory=asyncio.Lock)
    browser: Optional[BrowserUseBrowser] = Field(default=None, exclude=True)
    context: Optional[BrowserContext] = Field(default=None, exclude=True)
    dom_service: Optional[DomService] = Field(default=None, exclude=True)
    web_search_tool: WebSearch = Field(default_factory=WebSearch, exclude=True)

    # Context for generic functionality
    tool_context: Optional[Context] = Field(default=None, exclude=True)

    llm: Optional[LLM] = Field(default_factory=LLM)

    @field_validator("parameters", mode="before")
    def validate_parameters(cls, v: dict, info: ValidationInfo) -> dict:
        if not v:
            raise ValueError("Parameters cannot be empty")
        return v

    async def _ensure_browser_initialized(self) -> BrowserContext:
        """Ensure browser and context are initialized."""
        if self.browser is None:
            logger.info(
                "🚀 Initializing browser instance... (this may take 10-30 seconds)"
            )

            # Enhanced stealth browser configuration
            browser_config_kwargs = {
                "headless": False,  # Changed to True for automatic operation
                "disable_security": True,
                "extra_chromium_args": [
                    "--disable-blink-features=AutomationControlled",
                    "--exclude-switches=enable-automation",
                    "--disable-extensions-except=ublock",
                    "--disable-plugins-discovery",
                    "--no-first-run",
                    "--no-service-autorun",
                    "--password-store=basic",
                    "--disable-background-timer-throttling",
                    "--disable-renderer-backgrounding",
                    "--disable-backgrounding-occluded-windows",
                    "--disable-features=TranslateUI,BlinkGenPropertyTrees",
                    "--disable-ipc-flooding-protection",
                    "--enable-features=NetworkService,NetworkServiceLogging",
                    "--force-color-profile=srgb",
                    "--metrics-recording-only",
                    "--use-mock-keychain",
                    "--disable-web-security",
                    "--allow-running-insecure-content",
                    "--disable-features=VizDisplayCompositor",
                ],
            }

            if config.browser_config:
                from browser_use.browser.browser import ProxySettings

                # handle proxy settings.
                if config.browser_config.proxy and config.browser_config.proxy.server:
                    browser_config_kwargs["proxy"] = ProxySettings(
                        server=config.browser_config.proxy.server,
                        username=config.browser_config.proxy.username,
                        password=config.browser_config.proxy.password,
                    )

                browser_attrs = [
                    "headless",
                    "disable_security",
                    "extra_chromium_args",
                    "chrome_instance_path",
                    "wss_url",
                    "cdp_url",
                ]

                for attr in browser_attrs:
                    value = getattr(config.browser_config, attr, None)
                    if value is not None:
                        if not isinstance(value, list) or value:
                            browser_config_kwargs[attr] = value

            logger.info("📦 Creating browser instance with stealth configuration...")
            self.browser = BrowserUseBrowser(BrowserConfig(**browser_config_kwargs))
            logger.info("✅ Browser instance created successfully")

        if self.context is None:
            logger.info("🌐 Creating browser context... (this may take 5-15 seconds)")

            # Create basic context configuration first
            context_config = BrowserContextConfig()

            # if there is context config in the config, use it.
            if (
                config.browser_config
                and hasattr(config.browser_config, "new_context_config")
                and config.browser_config.new_context_config
            ):
                context_config = config.browser_config.new_context_config

            # Create the context with basic config
            self.context = await self.browser.new_context(context_config)
            logger.info("✅ Browser context created successfully")

            # Now manually configure stealth settings after context creation
            logger.info("🔧 Configuring stealth settings and viewport...")
            page = await self.context.get_current_page()

            # Set viewport manually
            await page.set_viewport_size({"width": 1280, "height": 800})

            # Override user agent directly on the page
            await page.set_extra_http_headers(
                {
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
                }
            )

            # Add comprehensive stealth script to prevent automation detection
            logger.info("🛡️ Adding stealth scripts to prevent detection...")
            await page.add_init_script(
                """
                // Hide webdriver property
                Object.defineProperty(navigator, 'webdriver', { get: () => undefined });

                // Spoof plugins with realistic data
                Object.defineProperty(navigator, 'plugins', {
                    get: () => ({
                        length: 3,
                        0: { name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer' },
                        1: { name: 'Chrome PDF Viewer', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai' },
                        2: { name: 'Native Client', filename: 'internal-nacl-plugin' }
                    })
                });

                // Spoof languages
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['en-US', 'en']
                });

                // Add realistic chrome object
                window.chrome = {
                    runtime: {},
                    loadTimes: function() { return { firstPaintTime: Date.now() / 1000 }; },
                    csi: function() { return { onloadT: Date.now() }; }
                };

                // Override permission query to avoid detection
                const originalQuery = window.navigator.permissions.query;
                window.navigator.permissions.query = (parameters) => (
                    parameters.name === 'notifications' ?
                        Promise.resolve({ state: Notification.permission }) :
                        originalQuery(parameters)
                );

                // Mock realistic screen properties
                Object.defineProperty(screen, 'availWidth', { get: () => 1440 });
                Object.defineProperty(screen, 'availHeight', { get: () => 900 });
                Object.defineProperty(screen, 'width', { get: () => 1440 });
                Object.defineProperty(screen, 'height', { get: () => 900 });

                // Hide automation indicators
                delete window.cdc_adoQpoasnfa76pfcZLmcfl_Array;
                delete window.cdc_adoQpoasnfa76pfcZLmcfl_Promise;
                delete window.cdc_adoQpoasnfa76pfcZLmcfl_Symbol;
                delete window.cdc_adoQpoasnfa76pfcZLmcfl_Object;
                delete window.cdc_adoQpoasnfa76pfcZLmcfl_Proxy;

                // Mock realistic battery API
                if (navigator.getBattery) {
                    navigator.getBattery = () => Promise.resolve({
                        charging: true,
                        chargingTime: 0,
                        dischargingTime: Infinity,
                        level: 1
                    });
                }
            """
            )

            self.dom_service = DomService(page)

            # If using CDP connection, ensure we're not stuck on the default playwright page
            try:
                current_url = page.url
                if (
                    "playwright" in current_url.lower()
                    or current_url.startswith("about:")
                    or "chrome-error://" in current_url
                ):
                    logger.info("🔄 Navigating to blank page to ensure fresh start...")
                    # Navigate to a blank page to ensure we start fresh
                    await page.goto(
                        "about:blank", wait_until="domcontentloaded", timeout=10000
                    )
                    await asyncio.sleep(1)  # Give it a moment to settle
            except Exception as e:
                # If there's any issue with the initial page, just continue
                # This can happen with CDP connections
                logger.warning(f"⚠️ Minor issue during initial page setup: {e}")
                pass

            logger.info("✅ Browser initialization completed successfully!")

        return self.context

    async def execute(
        self,
        action: str,
        url: Optional[str] = None,
        index: Optional[int] = None,
        text: Optional[str] = None,
        scroll_amount: Optional[int] = None,
        tab_id: Optional[int] = None,
        query: Optional[str] = None,
        goal: Optional[str] = None,
        keys: Optional[str] = None,
        seconds: Optional[int] = None,
        **kwargs,
    ) -> ToolResult:
        """
        Execute a specified browser action.

        Args:
            action: The browser action to perform
            url: URL for navigation or new tab
            index: Element index for click or input actions
            text: Text for input action or search query
            scroll_amount: Pixels to scroll for scroll action
            tab_id: Tab ID for switch_tab action
            query: Search query for Google search
            goal: Extraction goal for content extraction
            keys: Keys to send for keyboard actions
            seconds: Seconds to wait
            **kwargs: Additional arguments

        Returns:
            ToolResult with the action's output or error
        """
        async with self.lock:
            try:
                context = await self._ensure_browser_initialized()

                # Get max content length from config
                max_content_length = getattr(
                    config.browser_config, "max_content_length", 2000
                )

                # Navigation actions
                if action == "go_to_url":
                    if not url:
                        return ToolResult(
                            error="URL is required for 'go_to_url' action"
                        )
                    page = await context.get_current_page()

                    # Bring page to front first
                    await page.bring_to_front()

                    # Navigate with proper waiting
                    await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                    await page.wait_for_load_state("domcontentloaded")

                    # Enhanced wait for dynamic content and JavaScript-heavy pages
                    await asyncio.sleep(3)  # Increased wait time

                    # Additional wait for JavaScript frameworks to load
                    try:
                        await page.wait_for_load_state("networkidle", timeout=10000)
                    except:
                        pass  # Continue if networkidle wait fails

                    # Check for redirects to blocked/empty pages
                    final_url = page.url

                    # Enhanced redirect detection including Google Tag Manager and other tracking redirects
                    problematic_redirects = [
                        "googletagmanager.com",
                        "google-analytics.com",
                        "doubleclick.net",
                        "googlesyndication.com",
                        "analytics.google.com",
                        "pinterest.com/ct.html",
                        "facebook.com/tr",
                        "twitter.com/i/adsct",
                        "linkedin.com/li/track",
                        "adsystem.amazon.com",
                        "scorecardresearch.com",
                        "bing.com/cl/act",
                    ]

                    is_problematic_redirect = any(
                        domain in final_url.lower() for domain in problematic_redirects
                    )

                    if (
                        final_url.startswith("about:")
                        or "blank" in final_url.lower()
                        or (final_url != url and "error" in final_url.lower())
                        or is_problematic_redirect
                    ):

                        # Specific handling for tracking/analytics redirects
                        if is_problematic_redirect:
                            return ToolResult(
                                output=f"❌ Navigation to {url} was redirected to tracking/analytics service ({final_url}). "
                                f"This indicates the site is likely blocking automated access or the page doesn't exist. "
                                f"Do not retry - consider using web search to find the information instead.",
                                error="Redirect to tracking service detected",
                            )

                        # Site likely blocked us - suggest alternative approach
                        elif "youtube.com" in url.lower() or "video" in url.lower():
                            return ToolResult(
                                output=f"⚠️ Direct access to {url} was blocked (redirected to {final_url}). "
                                f"Consider using web_search to find video information instead.",
                                error="Site blocked direct access",
                            )
                        else:
                            return ToolResult(
                                output=f"⚠️ Navigation to {url} was redirected to {final_url}. "
                                f"Site may be blocking automated access. Consider alternative approaches.",
                                error="Redirect to blocked page detected",
                            )

                    # ENHANCED CONTENT VALIDATION
                    try:
                        # Get the page content for validation
                        page_content = await page.content()
                        page_title = await page.title()

                        # WHITELIST: Skip error detection for known good sites
                        trusted_domains = [
                            "google.com/search",
                            "bing.com/search",
                            "duckduckgo.com",
                            "search.yahoo.com",
                            "wikipedia.org",
                            "github.com",
                            "stackoverflow.com",
                        ]

                        is_trusted_site = any(
                            domain in final_url.lower() for domain in trusted_domains
                        )

                        if is_trusted_site:
                            # For trusted sites, use specific validation instead of generic error patterns
                            if "google.com/search" in final_url.lower():
                                # Validate Google search results page specifically
                                has_search_results = await page.evaluate(
                                    """
                                    () => {
                                        return document.querySelector('div#search, div#rso, div.g') !== null;
                                    }
                                """
                                )

                                if has_search_results:
                                    search_results_count = await page.evaluate(
                                        """
                                        () => {
                                            return document.querySelectorAll('div.g, div.yuRUbf').length;
                                        }
                                    """
                                    )

                                    return ToolResult(
                                        output=f"✅ Successfully loaded Google search results for query in URL. "
                                        f"Found {search_results_count} search result elements. Page title: '{page_title}'"
                                    )
                                else:
                                    # Google search page but no results - could be CAPTCHA or blocking
                                    captcha_detected = await page.evaluate(
                                        """
                                        () => {
                                            return document.querySelector('[src*="captcha"], #captcha, .captcha') !== null;
                                        }
                                    """
                                    )

                                    if captcha_detected:
                                        return ToolResult(
                                            output=f"⚠️ Google search loaded but shows CAPTCHA challenge. "
                                            f"This indicates rate limiting or bot detection. Try again later or use a different approach.",
                                            error="CAPTCHA challenge detected",
                                        )

                            # For other trusted sites, do minimal validation
                            visible_text = await page.evaluate(
                                """
                                () => document.body.innerText.trim()
                            """
                            )

                            if len(visible_text) > 100:
                                return ToolResult(
                                    output=f"✅ Successfully navigated to trusted site {final_url}. "
                                    f"Page title: '{page_title}'. Content appears valid."
                                )

                        # For non-trusted sites, apply full error detection
                        content_lower = page_content.lower()
                        title_lower = page_title.lower() if page_title else ""

                        # Error patterns that indicate the page failed to load properly
                        error_patterns = [
                            "page not found",
                            "404",
                            "error 404",
                            "access denied",
                            "forbidden",
                            "503 service unavailable",
                            "502 bad gateway",
                            "500 internal server error",
                            "this page doesn't exist",
                            "page does not exist",
                            "the page you requested could not be found",
                            "the requested url was not found",
                            "sorry, this page isn't available",
                            "oops! that page can't be found",
                            "we can't find that page",
                        ]

                        # CAPTCHA and security challenge patterns (enhanced for Cloudflare)
                        captcha_patterns = [
                            "verify you are human",
                            "complete the action below",
                            "security of your connection",
                            "review the security",
                            "prove you are human",
                            "solve the puzzle",
                            "captcha",
                            "i'm not a robot",
                            "security check",
                            "verification required",
                            "human verification",
                            "cloudflare",
                            "checking your browser",
                            "please wait while we validate",
                            "ddos protection",
                            "anti-bot verification",
                            "bot protection",
                            "browser verification",
                            "challenge verification",
                            "ray id",  # Cloudflare specific
                            "challenge not passed",
                            "enable javascript and cookies",
                            "browser integrity check",
                            "attention required",
                            "cloudflare ray id",
                        ]

                        # Check if any error patterns are present (but be more careful about "not found")
                        has_error_content = False
                        for pattern in error_patterns:
                            if pattern in title_lower:
                                has_error_content = True
                                break
                            # For content, be more strict - avoid false positives from search results
                            if pattern in content_lower and pattern != "not found":
                                has_error_content = True
                                break
                            # Only flag "not found" if it appears in very specific error contexts
                            if pattern == "not found" and any(
                                error_context in content_lower
                                for error_context in [
                                    "the page you requested could not be found",
                                    "this page could not be found",
                                    "sorry, the page you requested was not found",
                                    "error: page not found",
                                ]
                            ):
                                has_error_content = True
                                break

                        # Check for CAPTCHA or security challenges (but not on tracking domains)
                        tracking_domains = [
                            "doubleclick.net",
                            "googletagmanager.com",
                            "google-analytics.com",
                            "pinterest.com",
                            "facebook.com",
                            "twitter.com",
                            "linkedin.com",
                            "googlesyndication.com",
                            "adsystem.amazon.com",
                            "scorecardresearch.com",
                        ]

                        is_tracking_domain = any(
                            domain in final_url.lower() for domain in tracking_domains
                        )

                        # Only check for CAPTCHA on non-tracking domains
                        has_captcha_content = False
                        if not is_tracking_domain:
                            has_captcha_content = any(
                                pattern in content_lower or pattern in title_lower
                                for pattern in captcha_patterns
                            )

                            # Additional validation: ensure it's really a CAPTCHA page
                            if has_captcha_content:
                                # Check for actual CAPTCHA elements in the DOM
                                captcha_elements = await page.evaluate(
                                    """
                                    () => {
                                        const captchaSelectors = [
                                            '[class*="captcha"]', '[id*="captcha"]',
                                            '[class*="recaptcha"]', '[id*="recaptcha"]',
                                            'iframe[src*="recaptcha"]', 'iframe[src*="captcha"]',
                                            '[class*="cloudflare"]', '[id*="cf-challenge"]'
                                        ];
                                        return captchaSelectors.some(selector =>
                                            document.querySelector(selector) !== null
                                        );
                                    }
                                """
                                )

                                # Only flag as CAPTCHA if both text patterns AND DOM elements are present
                                has_captcha_content = captcha_elements

                        if has_captcha_content:
                            return ToolResult(
                                output=f"🤖 CAPTCHA/Security Challenge detected on {url}. "
                                f"Page title: '{page_title}'. "
                                f"The site is requiring human verification which cannot be automated. "
                                f"**CRITICAL**: Do not retry this URL - CAPTCHA pages will always block automation. "
                                f"Try alternative approaches: web search, different sites, or manual verification.",
                                error="CAPTCHA challenge detected - automation blocked",
                            )

                        # Check for minimal content (likely broken or empty page)
                        visible_text = await page.evaluate(
                            """
                            () => {
                                // Get all visible text content
                                const walker = document.createTreeWalker(
                                    document.body,
                                    NodeFilter.SHOW_TEXT,
                                    null,
                                    false
                                );
                                let text = '';
                                let node;
                                while (node = walker.nextNode()) {
                                    const parent = node.parentElement;
                                    if (parent && getComputedStyle(parent).display !== 'none' &&
                                        getComputedStyle(parent).visibility !== 'hidden') {
                                        text += node.textContent + ' ';
                                    }
                                }
                                return text.trim();
                            }
                        """
                        )

                        visible_text_length = (
                            len(visible_text.strip()) if visible_text else 0
                        )

                        # Check for JavaScript-heavy pages that might not have loaded
                        body_content = await page.evaluate(
                            "() => document.body.innerHTML"
                        )
                        has_meaningful_structure = len(body_content.strip()) > 500

                        # Determine if the page loaded successfully
                        if has_error_content:
                            return ToolResult(
                                output=f"❌ Page at {url} loaded but shows an error. "
                                f"Title: '{page_title}'. Content indicates: Page not found or access denied. "
                                f"Do not retry - the URL may be invalid or the content may not exist.",
                                error="Page loaded but contains error content",
                            )

                        if visible_text_length < 50 and not has_meaningful_structure:
                            return ToolResult(
                                output=f"❌ Page at {url} loaded but appears to be empty or broken. "
                                f"Title: '{page_title}'. Visible text length: {visible_text_length} chars. "
                                f"This may indicate a JavaScript-heavy page that failed to load, blocked content, or invalid URL. "
                                f"Do not retry - consider using web search instead.",
                                error="Page loaded but contains insufficient content",
                            )

                        # Additional check for pages that loaded but have no useful navigation elements
                        interactive_elements_count = await page.evaluate(
                            """
                            () => {
                                const clickables = document.querySelectorAll('a, button, input[type="submit"], [role="button"], [onclick]');
                                return clickables.length;
                            }
                        """
                        )

                        if visible_text_length < 200 and interactive_elements_count < 3:
                            return ToolResult(
                                output=f"⚠️ Page at {url} loaded but has minimal content. "
                                f"Title: '{page_title}'. Visible text: {visible_text_length} chars, "
                                f"Interactive elements: {interactive_elements_count}. "
                                f"This may be a landing page or the specific content may not exist. "
                                f"Consider searching for the content instead.",
                                error="Page loaded but has minimal useful content",
                            )

                        # Page appears to have loaded successfully with meaningful content
                        return ToolResult(
                            output=f"✅ Successfully navigated to {url} (final URL: {final_url}). "
                            f"Page title: '{page_title}'. Content appears valid with {visible_text_length} chars of text "
                            f"and {interactive_elements_count} interactive elements."
                        )

                    except Exception as content_check_error:
                        # If content validation fails, provide a warning but don't fail completely
                        return ToolResult(
                            output=f"⚠️ Navigated to {url} but content validation failed: {str(content_check_error)}. "
                            f"Page may have loaded but content checking encountered an error. Proceed with caution.",
                            error="Content validation failed",
                        )

                elif action == "go_back":
                    page = await context.get_current_page()

                    # Check if we can actually go back
                    try:
                        # Use page.go_back() with increased timeout and better error handling
                        await page.go_back(wait_until="domcontentloaded", timeout=30000)
                        # Additional wait for any dynamic content
                        await asyncio.sleep(1)
                        return ToolResult(output="Navigated back")
                    except Exception as e:
                        # If can't go back, check the current URL and provide helpful info
                        current_url = page.url
                        if (
                            "newtab" in current_url.lower()
                            or current_url.startswith("about:")
                            or current_url.startswith("chrome://")
                            or "blank" in current_url.lower()
                        ):
                            return ToolResult(
                                output="Cannot go back - currently on a new tab or about page. Consider navigating to a specific URL instead.",
                                error="No history to go back to",
                            )
                        elif "timeout" in str(e).lower():
                            # Try alternative: refresh the current page instead
                            try:
                                await page.reload(
                                    wait_until="domcontentloaded", timeout=15000
                                )
                                return ToolResult(
                                    output=f"Go back timed out, refreshed current page instead. Now at: {page.url}"
                                )
                            except:
                                return ToolResult(
                                    output=f"Go back failed with timeout from {current_url}. Consider navigating to a specific URL.",
                                    error="Navigation timeout - try direct URL navigation",
                                )
                        else:
                            return ToolResult(
                                output=f"Could not navigate back from {current_url}: {str(e)}",
                                error=f"Navigation failed: {str(e)}",
                            )

                elif action == "refresh":
                    await context.refresh_page()
                    return ToolResult(output="Refreshed current page")

                elif action == "web_search":
                    if not query:
                        return ToolResult(
                            error="Query is required for 'web_search' action"
                        )

                    try:
                        page = await context.get_current_page()

                        # Bring page to front and ensure it's ready
                        await page.bring_to_front()

                        # Try direct search URL first (more reliable)
                        import urllib.parse

                        search_url = f"https://www.google.com/search?q={urllib.parse.quote_plus(query)}"

                        try:
                            await page.goto(
                                search_url, wait_until="domcontentloaded", timeout=30000
                            )

                            # Wait for search results to load
                            await page.wait_for_selector("div#search", timeout=10000)
                            await asyncio.sleep(1)  # Allow for any dynamic content

                            final_url = page.url
                            if "google.com" in final_url:
                                return ToolResult(
                                    output=f"Successfully performed Google search for '{query}'. Search results are now loaded and visible."
                                )
                            else:
                                return ToolResult(
                                    output=f"Search completed but redirected to: {final_url}"
                                )

                        except Exception as direct_error:
                            # Fallback: Navigate to Google homepage and search manually
                            await page.goto(
                                "https://www.google.com",
                                wait_until="domcontentloaded",
                                timeout=15000,
                            )

                            # Wait for and find the search input
                            try:
                                await page.wait_for_selector(
                                    "input[name='q'], input[title*='Search'], textarea[name='q']",
                                    timeout=10000,
                                )

                                # Fill the search box
                                search_input = page.locator(
                                    "input[name='q'], input[title*='Search'], textarea[name='q']"
                                ).first
                                await search_input.fill(query)

                                # Submit the search
                                await search_input.press("Enter")

                                # Wait for results page
                                await page.wait_for_load_state(
                                    "domcontentloaded", timeout=15000
                                )
                                await asyncio.sleep(2)

                                return ToolResult(
                                    output=f"Successfully searched for '{query}' using Google homepage. Search results should now be visible."
                                )

                            except Exception as manual_error:
                                return ToolResult(
                                    error=f"Both direct and manual search failed. Direct: {direct_error}, Manual: {manual_error}"
                                )

                    except Exception as e:
                        return ToolResult(
                            error=f"Failed to perform web search: {str(e)}"
                        )

                elif action == "search_video_content":
                    if not query:
                        return ToolResult(
                            error="Query is required for 'search_video_content' action"
                        )

                    try:
                        page = await context.get_current_page()
                        await page.bring_to_front()

                        # Use DuckDuckGo videos as it's less likely to block automation
                        import urllib.parse

                        video_search_url = f"https://duckduckgo.com/?q={urllib.parse.quote_plus(query)}&ia=videos"

                        try:
                            await page.goto(
                                video_search_url,
                                wait_until="domcontentloaded",
                                timeout=30000,
                            )

                            # Wait for video results to load
                            await page.wait_for_selector(
                                ".tile--vid, .results", timeout=10000
                            )
                            await asyncio.sleep(2)

                            # Take screenshot of results
                            screenshot = await page.screenshot(
                                full_page=False, type="jpeg", quality=80
                            )
                            screenshot_b64 = base64.b64encode(screenshot).decode(
                                "utf-8"
                            )

                            final_url = page.url
                            return ToolResult(
                                output=f"Successfully searched for video content: '{query}'. Video search results are now displayed. Use extract_content to gather specific video information or click on video links to access them.",
                                base64_image=screenshot_b64,
                            )

                        except Exception as duckduckgo_error:
                            # Fallback to Google video search
                            try:
                                google_video_url = f"https://www.google.com/search?q={urllib.parse.quote_plus(query)}&tbm=vid"
                                await page.goto(
                                    google_video_url,
                                    wait_until="domcontentloaded",
                                    timeout=30000,
                                )

                                await page.wait_for_selector(
                                    "div#search, .g", timeout=10000
                                )
                                await asyncio.sleep(2)

                                screenshot = await page.screenshot(
                                    full_page=False, type="jpeg", quality=80
                                )
                                screenshot_b64 = base64.b64encode(screenshot).decode(
                                    "utf-8"
                                )

                                return ToolResult(
                                    output=f"Successfully searched for video content: '{query}' using Google Videos. Video search results are displayed.",
                                    base64_image=screenshot_b64,
                                )

                            except Exception as google_error:
                                return ToolResult(
                                    error=f"Both DuckDuckGo and Google video search failed. DuckDuckGo: {duckduckgo_error}, Google: {google_error}"
                                )

                    except Exception as e:
                        return ToolResult(
                            error=f"Failed to perform video content search: {str(e)}"
                        )

                # Element interaction actions
                elif action == "click_element":
                    if index is None:
                        return ToolResult(
                            error="Index is required for 'click_element' action"
                        )
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")

                    # Safety check: avoid clicking problematic elements
                    element_text = getattr(element, "text", "").lower()
                    element_href = getattr(element, "href", "") or ""

                    # Check for problematic links/elements
                    problematic_patterns = [
                        "sign in",
                        "sign up",
                        "login",
                        "register",
                        "privacy",
                        "terms",
                        "cookie",
                        "policy",
                        "help",
                        "support",
                        "contact",
                        "about",
                    ]

                    if any(pattern in element_text for pattern in problematic_patterns):
                        return ToolResult(
                            output=f"⚠️ Skipped clicking element {index} ('{element_text}') - appears to be navigation/auth link. "
                            f"Consider clicking on main content results instead.",
                            error="Avoided clicking problematic element",
                        )

                    if (
                        "accounts.google.com" in element_href
                        or "signin" in element_href.lower()
                    ):
                        return ToolResult(
                            output=f"⚠️ Skipped clicking element {index} - Google sign-in link detected. "
                            f"Focus on main search results instead.",
                            error="Avoided Google sign-in redirect",
                        )

                    # Proceed with click if element seems safe
                    download_path = await context._click_element_node(element)
                    output = f"Clicked element at index {index}"
                    if download_path:
                        output += f" - Downloaded file to {download_path}"
                    return ToolResult(output=output)

                elif action == "input_text":
                    if index is None or not text:
                        return ToolResult(
                            error="Index and text are required for 'input_text' action"
                        )
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")
                    await context._input_text_element_node(element, text)
                    return ToolResult(
                        output=f"Input '{text}' into element at index {index}"
                    )

                elif action == "scroll_down" or action == "scroll_up":
                    direction = 1 if action == "scroll_down" else -1
                    amount = (
                        scroll_amount
                        if scroll_amount is not None
                        else context.config.browser_window_size["height"]
                    )
                    await context.execute_javascript(
                        f"window.scrollBy(0, {direction * amount});"
                    )
                    return ToolResult(
                        output=f"Scrolled {'down' if direction > 0 else 'up'} by {amount} pixels"
                    )

                elif action == "scroll_to_text":
                    if not text:
                        return ToolResult(
                            error="Text is required for 'scroll_to_text' action"
                        )
                    page = await context.get_current_page()
                    try:
                        locator = page.get_by_text(text, exact=False)
                        await locator.scroll_into_view_if_needed()
                        return ToolResult(output=f"Scrolled to text: '{text}'")
                    except Exception as e:
                        return ToolResult(error=f"Failed to scroll to text: {str(e)}")

                elif action == "send_keys":
                    if not keys:
                        return ToolResult(
                            error="Keys are required for 'send_keys' action"
                        )
                    page = await context.get_current_page()
                    await page.keyboard.press(keys)
                    return ToolResult(output=f"Sent keys: {keys}")

                elif action == "get_dropdown_options":
                    if index is None:
                        return ToolResult(
                            error="Index is required for 'get_dropdown_options' action"
                        )
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")
                    page = await context.get_current_page()
                    options = await page.evaluate(
                        """
                        (xpath) => {
                            const select = document.evaluate(xpath, document, null,
                                XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                            if (!select) return null;
                            return Array.from(select.options).map(opt => ({
                                text: opt.text,
                                value: opt.value,
                                index: opt.index
                            }));
                        }
                    """,
                        element.xpath,
                    )
                    return ToolResult(output=f"Dropdown options: {options}")

                elif action == "select_dropdown_option":
                    if index is None or not text:
                        return ToolResult(
                            error="Index and text are required for 'select_dropdown_option' action"
                        )
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")
                    page = await context.get_current_page()
                    await page.select_option(element.xpath, label=text)
                    return ToolResult(
                        output=f"Selected option '{text}' from dropdown at index {index}"
                    )

                # Content extraction actions
                elif action == "extract_content":
                    if not goal:
                        return ToolResult(
                            error="Goal is required for 'extract_content' action"
                        )

                    page = await context.get_current_page()

                    # First check if we're on a valid page
                    current_url = page.url
                    if (
                        current_url.startswith("about:")
                        or "blank" in current_url.lower()
                        or "error" in current_url.lower()
                    ):
                        return ToolResult(
                            error=f"Cannot extract content from {current_url} - page appears to be blank or blocked"
                        )

                    # Get page title and basic info
                    try:
                        title = await page.title()
                        url_info = f"Page: {title} ({current_url})\n\n"
                    except:
                        url_info = f"Page: {current_url}\n\n"

                    # Take a screenshot for context
                    screenshot_b64 = None
                    try:
                        screenshot = await page.screenshot(
                            full_page=False, type="jpeg", quality=80
                        )
                        screenshot_b64 = base64.b64encode(screenshot).decode("utf-8")
                    except:
                        pass

                    import markdownify

                    # Enhanced content extraction for JavaScript-heavy pages
                    try:
                        # Wait a bit more for any remaining JavaScript to finish
                        await asyncio.sleep(2)

                        # Try to scroll to trigger lazy loading
                        await page.evaluate(
                            "window.scrollTo(0, document.body.scrollHeight / 2)"
                        )
                        await asyncio.sleep(1)
                        await page.evaluate("window.scrollTo(0, 0)")
                        await asyncio.sleep(1)

                        # Get both innerHTML and rendered text
                        page_html = await page.content()
                        visible_text = await page.evaluate(
                            """
                            () => {
                                // Get all visible text content more comprehensively
                                const elements = document.querySelectorAll('*');
                                let text = '';
                                elements.forEach(el => {
                                    if (el.offsetParent !== null && el.innerText && el.innerText.trim()) {
                                        const style = getComputedStyle(el);
                                        if (style.display !== 'none' && style.visibility !== 'hidden') {
                                            text += el.innerText + '\\n';
                                        }
                                    }
                                });
                                return text;
                            }
                        """
                        )

                        # Use the richer content source
                        if visible_text and len(visible_text.strip()) > len(
                            markdownify.markdownify(page_html).strip()
                        ):
                            content = visible_text
                        else:
                            content = markdownify.markdownify(page_html)

                    except Exception as extract_error:
                        logger.warning(
                            f"Enhanced extraction failed: {extract_error}, falling back to basic extraction"
                        )
                        content = markdownify.markdownify(await page.content())

                    # ENHANCED CONTENT VALIDATION for extraction
                    if len(content.strip()) < 100:
                        return ToolResult(
                            error=f"Cannot extract content from {current_url} - page contains insufficient content ({len(content)} chars). "
                            f"The page may be empty, broken, or still loading. Do not retry extraction."
                        )

                    # Check for error content in the page
                    content_lower = content.lower()
                    error_indicators = [
                        "page not found",
                        "404",
                        "error 404",
                        "not found",
                        "access denied",
                        "forbidden",
                        "this page doesn't exist",
                        "sorry, this page isn't available",
                        "oops! that page can't be found",
                    ]

                    if any(
                        indicator in content_lower for indicator in error_indicators
                    ):
                        return ToolResult(
                            error=f"Cannot extract content from {current_url} - page shows error content. "
                            f"The requested content likely does not exist. Do not retry extraction."
                        )

                    # Truncate content for processing
                    truncated_content = content[:max_content_length]

                    # First try with LLM extraction, but with timeout protection
                    try:
                        prompt = f"""\
Your task is to extract the content of the page. You will be given a page and a goal, and you should extract all relevant information around this goal from the page. If the goal is vague, summarize the page. Respond in json format.
Extraction goal: {goal}

Page content:
{truncated_content}
"""
                        messages = [{"role": "system", "content": prompt}]

                        # Define extraction function schema
                        extraction_function = {
                            "type": "function",
                            "function": {
                                "name": "extract_content",
                                "description": "Extract specific information from a webpage based on a goal",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "extracted_content": {
                                            "type": "object",
                                            "description": "The content extracted from the page according to the goal",
                                            "properties": {
                                                "text": {
                                                    "type": "string",
                                                    "description": "Text content extracted from the page",
                                                },
                                                "metadata": {
                                                    "type": "object",
                                                    "description": "Additional metadata about the extracted content",
                                                    "properties": {
                                                        "source": {
                                                            "type": "string",
                                                            "description": "Source of the extracted content",
                                                        }
                                                    },
                                                },
                                            },
                                        }
                                    },
                                    "required": ["extracted_content"],
                                },
                            },
                        }

                        # Use LLM to extract content with timeout protection
                        response = await asyncio.wait_for(
                            self.llm.ask_tool(
                                messages,
                                tools=[extraction_function],
                                tool_choice="required",
                                timeout=60,  # Shorter timeout for nested LLM call
                            ),
                            timeout=90,  # Overall timeout for this operation
                        )

                        if response and response.tool_calls:
                            args = json.loads(response.tool_calls[0].function.arguments)
                            extracted_content = args.get("extracted_content", {})
                            return ToolResult(
                                output=f"Extracted from page:\n{extracted_content}\n"
                            )

                    except asyncio.TimeoutError:
                        # LLM extraction timed out, fall back to simple text extraction
                        pass
                    except Exception as e:
                        # LLM extraction failed, fall back to simple text extraction
                        pass

                    # Fallback: Simple text extraction without LLM
                    # Extract text relevant to the goal using basic text processing
                    lines = truncated_content.split("\n")
                    relevant_lines = []
                    goal_keywords = goal.lower().split()

                    for line in lines:
                        line_lower = line.lower()
                        if any(keyword in line_lower for keyword in goal_keywords):
                            relevant_lines.append(line.strip())

                    if relevant_lines:
                        extracted_text = "\n".join(
                            relevant_lines[:20]
                        )  # Limit to first 20 relevant lines
                    else:
                        # If no relevant lines found, return first part of content
                        extracted_text = "\n".join(
                            lines[:10]
                        )  # First 10 lines as fallback

                    # Combine with page info and screenshot
                    result_output = f"{url_info}Extraction Goal: {goal}\n\nExtracted Content:\n{extracted_text}\n"

                    # Save debug output
                    await self._save_debug_output(
                        "extract_content", result_output, screenshot_b64
                    )

                    return ToolResult(output=result_output, base64_image=screenshot_b64)

                # Tab management actions
                elif action == "switch_tab":
                    if tab_id is None:
                        return ToolResult(
                            error="Tab ID is required for 'switch_tab' action"
                        )
                    await context.switch_to_tab(tab_id)
                    page = await context.get_current_page()
                    await page.wait_for_load_state()
                    return ToolResult(output=f"Switched to tab {tab_id}")

                elif action == "open_tab":
                    if not url:
                        return ToolResult(error="URL is required for 'open_tab' action")
                    await context.create_new_tab(url)
                    return ToolResult(output=f"Opened new tab with {url}")

                elif action == "close_tab":
                    await context.close_current_tab()
                    return ToolResult(output="Closed current tab")

                # Utility actions
                elif action == "wait":
                    seconds_to_wait = seconds if seconds is not None else 3
                    await asyncio.sleep(seconds_to_wait)
                    return ToolResult(output=f"Waited for {seconds_to_wait} seconds")

                else:
                    return ToolResult(error=f"Unknown action: {action}")

            except Exception as e:
                return ToolResult(error=f"Browser action '{action}' failed: {str(e)}")

    async def get_current_state(
        self, context: Optional[BrowserContext] = None
    ) -> ToolResult:
        """
        Get the current browser state as a ToolResult.
        If context is not provided, uses self.context.
        """
        try:
            # Use provided context or fall back to self.context
            ctx = context or self.context
            if not ctx:
                return ToolResult(error="Browser context not initialized")

            state = await ctx.get_state()

            # Create a viewport_info dictionary if it doesn't exist
            viewport_height = 0
            if hasattr(state, "viewport_info") and state.viewport_info:
                viewport_height = state.viewport_info.height
            elif hasattr(ctx, "config") and hasattr(ctx.config, "browser_window_size"):
                viewport_height = ctx.config.browser_window_size.get("height", 0)

            # Take a screenshot for the state
            page = await ctx.get_current_page()

            await page.bring_to_front()
            await page.wait_for_load_state("domcontentloaded")

            # Brief wait for any dynamic content
            await asyncio.sleep(1)

            # Get current URL to inform screenshot context
            current_url = page.url
            url_context = f" (Current URL: {current_url})"

            # DEFENSIVE CHECK: Avoid screenshots of problematic pages
            if (
                current_url.startswith("about:")
                or "blank" in current_url.lower()
                or "error" in current_url.lower()
                or current_url.startswith("chrome://")
                or current_url.startswith("chrome-error://")
            ):
                return ToolResult(
                    error=f"Cannot get browser state from {current_url} - page is blank, blocked, or shows an error. "
                    f"Navigate to a valid URL before requesting browser state."
                )

            # Take screenshot with error handling
            try:
                screenshot = await page.screenshot(
                    full_page=False,  # Use viewport screenshot for faster performance
                    animations="disabled",
                    type="jpeg",
                    quality=85,
                )
                screenshot_b64 = base64.b64encode(screenshot).decode("utf-8")
            except Exception as e:
                # If screenshot fails, create a simple placeholder
                logger.warning(f"Screenshot failed: {e}")
                screenshot_b64 = None

            # Build the state info with all required fields
            state_info = {
                "url": state.url,
                "title": state.title,
                "tabs": [tab.model_dump() for tab in state.tabs],
                "help": "[0], [1], [2], etc., represent clickable indices corresponding to the elements listed. Clicking on these indices will navigate to or interact with the respective content behind them.",
                "interactive_elements": (
                    state.element_tree.clickable_elements_to_string()
                    if state.element_tree
                    else ""
                ),
                "scroll_info": {
                    "pixels_above": getattr(state, "pixels_above", 0),
                    "pixels_below": getattr(state, "pixels_below", 0),
                    "total_height": getattr(state, "pixels_above", 0)
                    + getattr(state, "pixels_below", 0)
                    + viewport_height,
                },
                "viewport_height": viewport_height,
            }

            # Add URL context to the output
            enhanced_state_info = {
                **state_info,
                "current_state_context": f"Browser is currently showing: {state.title or 'Untitled'}{url_context}",
            }

            # Save debug output
            debug_content = json.dumps(
                enhanced_state_info, indent=4, ensure_ascii=False
            )
            await self._save_debug_output(
                "get_current_state", debug_content, screenshot_b64
            )

            return ToolResult(
                output=debug_content,
                base64_image=screenshot_b64,
            )
        except Exception as e:
            return ToolResult(error=f"Failed to get browser state: {str(e)}")

    async def cleanup(self):
        """Clean up browser resources."""
        async with self.lock:
            if self.context is not None:
                await self.context.close()
                self.context = None
                self.dom_service = None
            if self.browser is not None:
                await self.browser.close()
                self.browser = None

    async def _save_debug_output(
        self, action: str, content: str, screenshot_b64: Optional[str] = None
    ) -> None:
        """Save debug output to workspace folder for troubleshooting."""
        try:
            workspace_path = Path(config.workspace_root)
            debug_folder = workspace_path / "browser_debug"
            debug_folder.mkdir(exist_ok=True)

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save text content
            content_file = debug_folder / f"{timestamp}_{action}.txt"
            with open(content_file, "w", encoding="utf-8") as f:
                f.write(f"Action: {action}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Content:\n{content}\n")

            # Save screenshot if provided
            if screenshot_b64:
                screenshot_file = debug_folder / f"{timestamp}_{action}.jpg"
                screenshot_data = base64.b64decode(screenshot_b64)
                with open(screenshot_file, "wb") as f:
                    f.write(screenshot_data)

            print(f"🐛 Debug output saved to: {debug_folder}")
        except Exception as e:
            # Don't fail the main operation if debug saving fails
            print(f"⚠️ Failed to save debug output: {e}")

    def __del__(self):
        """Ensure cleanup when object is destroyed."""
        if self.browser is not None or self.context is not None:
            try:
                asyncio.run(self.cleanup())
            except RuntimeError:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self.cleanup())
                loop.close()

    @classmethod
    def create_with_context(cls, context: Context) -> "BrowserUseTool[Context]":
        """Factory method to create a BrowserUseTool with a specific context."""
        tool = cls()
        tool.tool_context = context
        return tool
