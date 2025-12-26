import re
import time

import numpy as np
from diskcache import Cache
from tqdm import tqdm

from data_set import DATA_SET

cache = Cache("~/PycharmProjects/tavily/.diskcache")

# Compiled regex patterns for better performance
_RE_IMAGE_LINK = re.compile(r'!\[(.*?)\]\([^\)]+\)')
_RE_MARKDOWN_LINK = re.compile(r'\[(.*?)\]\([^\)]+\)')
_RE_URL_PROTOCOL = re.compile(r'^https?://')
_RE_HAS_LETTERS = re.compile(r'[a-zA-Z\u00C0-\u024F\u1E00-\u1EFF]')
_RE_BARE_HTTP_URL = re.compile(r'(?<!\]\()https?://\s?[a-zA-Z0-9](?:[a-zA-Z0-9\-._~:/?#\[\]@!$&\'()*+,;=]*[a-zA-Z0-9])?')
_RE_WWW_DOMAIN = re.compile(r'[a-zA-Z]*www\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+(?:/[a-zA-Z0-9\-._~:/?#\[\]@!$&\'()*+,;=]*[a-zA-Z0-9])?')
_RE_MALFORMED_LINK = re.compile(r'([^\]]+)\]\([^\)]+\)')
_RE_MULTIPLE_NEWLINES = re.compile(r'\n{3,}')
_RE_HEADER_UNDERLINE_EQUALS = re.compile(r'={4,}')  # Reduce multiple = to ===
_RE_HEADER_UNDERLINE_DASHES = re.compile(r'-{4,}')  # Reduce multiple - to ---

# def _contains_non_relevant_word(line):
#     return bool(_non_relevant_pattern.search(line))

def process_markdown_link(match):
    """Process markdown links: handles empty, URL-as-text, citation, and regular links."""
    link_text = match.group(1).strip()
    
    # Remove empty links: [](url) or [ ](url)
    if not link_text:
        return ''
    
    # Remove links where text is a URL: [https://...](url) or [http://...](url)
    if _RE_URL_PROTOCOL.match(link_text):
        return ''
    
    # Remove citation links: text contains only numbers, brackets, or special characters
    # Examples: [\[1\]](url), [1](url), [[1]](url)
    unescaped_text = link_text.replace('\\', '')
    if not _RE_HAS_LETTERS.search(unescaped_text):
        return ''  # Remove citation links completely
    
    # Regular links with text: return just the text
    return link_text

@cache.memoize()
def get_clean_markdown_text_heading(md_text):
    """
    Optimized version: applies all regex patterns to entire text at once.
    No line-by-line processing - all done with global regex for maximum performance.
    """
    # Apply all regex patterns to the entire text at once (bulk processing)
    
    # Remove all image links
    md_text = _RE_IMAGE_LINK.sub('', md_text)
    
    # Remove zero-width spaces
    md_text = md_text.replace('\u200b', '')
    
    # Process all markdown links in one pass
    md_text = _RE_MARKDOWN_LINK.sub(process_markdown_link, md_text)
    
    # Remove bare HTTP/HTTPS URLs
    md_text = _RE_BARE_HTTP_URL.sub('', md_text)
    
    # Remove www domains (including concatenated words)
    md_text = _RE_WWW_DOMAIN.sub('', md_text)
    
    # Handle malformed links: text](url) -> text
    md_text = _RE_MALFORMED_LINK.sub(r'\1', md_text)
    
    # Reduce header underlines: multiple = to ===, multiple - to ---
    md_text = _RE_HEADER_UNDERLINE_EQUALS.sub('===', md_text)
    md_text = _RE_HEADER_UNDERLINE_DASHES.sub('---', md_text)
    
    # Collapse multiple consecutive blank lines
    md_text = _RE_MULTIPLE_NEWLINES.sub('\n\n', md_text)
    
    return md_text


input_md = """Latest security intelligence updates for Microsoft Defender Antivirus and other Microsoft antimalware - Microsoft Security Intelligence

===============
[Skip to main content](javascript:void(0))

![Image 1](https://www.microsoft.com/wdsi/definitions)

[Skip to main content](javascript:void(0))

[![Image 2](https://uhf.microsoft.com/images/microsoft/RE1Mu3b.png)Microsoft](https://www.microsoft.com/)

Microsoft Security Intelligence

[Microsoft Security Intelligence](https://www.microsoft.com/en-us/wdsi)

 Microsoft Security Intelligence 

*   [Home](https://www.microsoft.com/en-us/wdsi)
*   [Threats](https://www.microsoft.com/en-us/wdsi/threats)
*   [Blogs](https://cloudblogs.microsoft.com/microsoftsecure/?product=windows,windows-defender-advanced-threat-protection)

Manually download the update
----------------------------

You can manually download the latest update.

### Latest security intelligence update

The latest security intelligence update is:

- [x] Analytical and statistical cookies     

### QBI newsletters
[Subscribe](https://qbi.uq.edu.au/get-involved/qbi-newsletters)
[![The Brain: Intelligent Machines QBI](/files/40604/The-Brain-Intelligent-Machines-QBI-cover.jpg)](https://qbi.uq.edu.au/intelligentmachines?utm_source=IntelligentMachinesMag&utm_medium=Website&utm_campaign=SidebarSnippet)
[![The Brain: Intelligent Machines QBI](/files/40604/The-Brain-Intelligent-Machines-QBI-cover.jpg)](https://qbi.uq.edu.au/intelligentmachines?utm_source=IntelligentMachinesMag&utm_medium=Website&utm_campaign=SidebarSnippet)
![The Brain: Intelligent Machines QBI](/files/40604/The-Brain-Intelligent-Machines-QBI-cover.jpg)![Australian Aboriginal Flag](https://static.uq.net.au/v14/images/rap/aboriginal.svg)
![Torres Strait Islander Flag](https://static.uq.net.au/v14/images/rap/torres-strait-islanders.svg)
UQ acknowledges the Traditional Owners and their custodianship of the lands on which UQ is situated. —
[Reconciliation at UQ](https://about.uq.edu.au/reconciliation)
![Torres Strait Islander Flag](https://static.uq.net.au/v14/images/rap/torres-strait-islanders.svg)
UQ acknowledges the Traditional Owners and their custodianship of the lands on which UQ is situated. —
[Reconciliation at UQ](https://about.uq.edu.au/reconciliation)
## Media
https://www.nationalgeographic.com/animals/birds/facts/penguins-1
[https://www.nationalgeographic.com/animals/birds/facts/penguins-1](https://www.nationalgeographic.com/animals/birds/facts/penguins-1)
1. [Gumloop](https://www.gumloop.com/) (best for AI automations)
Gumloop is the most
asd](https://chat.deepseek.com)
体验全新旗舰模型](https://chat.deepseek.com)
*   www.iswi-secretariat.org Internationale Weltraumwetter-Initiative
[\[1\]](https://de.wikipedia.org/wiki/Weltraumwetter#cite_note-1)
・England, N. Your COVID Recovery/Returning to work. 2022 2022-06-30; Available from: http:// yourcovidrecovery.nhs.uk.
**[Artificial Intelligence (AI)](https://www.mckinsey.com/featured-insights/mckinsey-explainers/what-is-ai):**The ability of software to perform tasks that traditionally require human intelligence, mirroring some cognitive functions usually associated with human minds.
![Image 40: Read more about the article 15 Amazing Facts About African Penguins [#9 Might Make You Squirm]](https://polarguidebook.com/african-penguins-facts/)
NAICS categories do not distinguish between small and large business, or between for-profit and non-profit. The Small Business Administration (SBA) develops size standards for each NAICS category. To find more information about the SBA size standards, or when the SBA will update their size standards to reflect 2022 NAICS revisions, visit the SBA Web site atwww.sba.gov/federal-contracting/contracting-guide/size-standards. You may also contact SBA’s Office of Size Standards Answer Desk onor via email to[email protected]. [SOURCE: US NAICS MANUAL]"""

output_md = """Latest security intelligence updates for Microsoft Defender Antivirus and other Microsoft antimalware - Microsoft Security Intelligence
Microsoft
Microsoft Security Intelligence
Microsoft Security Intelligence
Microsoft Security Intelligence
*   Home
*   Threats
*   Blogs
Manually download the update
You can manually download the latest update.
Latest security intelligence update
The latest security intelligence update is:
QBI newsletters
Subscribe
UQ acknowledges the Traditional Owners and their custodianship of the lands on which UQ is situated. —
Reconciliation at UQ
UQ acknowledges the Traditional Owners and their custodianship of the lands on which UQ is situated. —
Reconciliation at UQ
Media
1. Gumloop (best for AI automations)
Gumloop is the most
asd
体验全新旗舰模型
*    Internationale Weltraumwetter-Initiative
・England, N. Your COVID Recovery/Returning to work. 2022 2022-06-30; Available from: .
**Artificial Intelligence (AI):**The ability of software to perform tasks that traditionally require human intelligence, mirroring some cognitive functions usually associated with human minds.
NAICS categories do not distinguish between small and large business, or between for-profit and non-profit. The Small Business Administration (SBA) develops size standards for each NAICS category. To find more information about the SBA size standards, or when the SBA will update their size standards to reflect 2022 NAICS revisions, visit the SBA Web site . You may also contact SBA’s Office of Size Standards Answer Desk onor via email to[email protected]. [SOURCE: US NAICS MANUAL]"""


def full_test():
    times = []
    for i, d in tqdm(enumerate(DATA_SET), total=len(DATA_SET)):
        md_text = d['markdown_content']
        start_time = time.time()
        clean_text = get_clean_markdown_text_heading(md_text)
        end_time = time.time()
        times.append(end_time - start_time)
        print(f'{np.min(times):.3f}, {np.mean(times):.3f}, {np.max(times):.3f}, {np.argmax(times):.3f}')
        if ('http' in clean_text or 'www' in clean_text) and i not in [6, 37, 16, 88]:
            print(i)
            print("Found URL in cleaned text:")
            print(clean_text)
            asd = 2
            clean_text1 = get_clean_markdown_text_heading(md_text)
        else:
            pass
            # print('all good')
            # print(clean_text)
            # asd = 3


if __name__ == "__main__":
    text = get_clean_markdown_text_heading(input_md)
    print("Expected:")
    print(output_md)
    print("\nActual:")
    print(text)
    print(text == output_md)
    full_test()