from __future__ import annotations
import re
import yt_dlp
from datetime import datetime, timedelta

YOUTUBE_URL_PATTERN = re.compile(
    r"(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w\-]+"
)

def is_youtube_url(url: str):
    return bool(YOUTUBE_URL_PATTERN.match(url.strip()))

def normalize_url(url:str):
    url=url.strip()
    if not url.startswith("http"):
        url="https://"+url
    return url

def calculate_engagement_score(video_info:dict):
    score = 0.0

    view_count = video_info.get("view_count",0) or 0
    if view_count > 0:
        import math
        score += math.log10(view_count+1)*10

    likes = video_info.get("like_count",0) or 0
    dislikes = video_info.get("dislike_count",0) or 0
    if likes + dislikes > 0:
        like_ratio = likes/(likes+dislikes) 
        score += like_ratio*20
    
    subs = video_info.get("channel_sub_count",0) or 0 
    if subs> 0:
        import math
        score+= math.log10(subs+1)*5

    upload_date= video_info.get("upload_date")
    if upload_date:
        try:
            upload_dt = datetime.strptime(str(upload_date), "%Y%m%d")
            days_old = (datetime.now() - upload_dt).days
            if days_old < 30:
                score += 15
            elif days_old < 90:
                score += 10
            elif days_old < 180:
                score += 5
        except:
            pass
    return score

def keyword_match(video_info:dict,query:str):
    title = (video_info.get("title") or "").lower()
    description = (video_info.get("description") or "").lower()
    keywords = query.lower().split()
    matches = sum(1 for kw in keywords if kw in title or kw in description)
    return matches >= len(keywords) * 0.5

def smart_discovery(
    query: str,
    max_results: int = 10,
    min_duration: int = 30,        # seconds
    max_duration: int = 3600,      # 1 hour
    sort_by: str = "relevance",    # "relevance", "view_count", "upload_date"
    require_keyword_match: bool = True
):
    print(f"[discovery] Smart search: '{query}'")
    print(f"[discovery] Filters: {min_duration}s-{max_duration}s duration, sort by {sort_by}")

    search_limit = max_results * 3

    sort_param = {
        "relevance":"",
        "view_count":"sp=CAMSAhAB",
        "upload_date":"sp=CAMSAhAB"
    }.get(sort_by,"")

    search_url = f"ytsearch{search_limit}:{query}"
    
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,  
        "skip_download": True,
    }

    videos = []

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(search_url,download=False)
            entries = info.get("entries",[])
            print(f"[discovery] Found {len(entries)} raw results, filtering...")

            for entry in entries:
                if not entry:
                    continue

                duration = entry.get("duration",0) or 0
                if duration < min_duration or duration > max_duration:
                    continue

                if require_keyword_match and not keyword_match(entry,query):
                    continue

                if entry.get("is_live"):
                    continue

                score = calculate_engagement_score(entry)

                vid_id = entry.get("id")
                if vid_id:
                     videos.append({
                        "url": f"https://www.youtube.com/watch?v={vid_id}",
                        "title": entry.get("title", "Unknown"),
                        "score": score,
                        "views": entry.get("view_count", 0),
                        "duration": duration,
                    })
                     
        except Exception as e:
            print(f"[discovery] Search error: {e}")
            return []
    
    videos.sort(key=lambda v: v["score"], reverse=True)

    top_videos = videos[:max_results]

    print(f"[discovery] Filtered to {len(top_videos)} high-quality results:")
    for i, v in enumerate(top_videos[:5], 1):  # Show top 5
        print(f"  {i}. {v['title'][:50]}... (score: {v['score']:.1f}, views: {v['views']:,})")
    
    if len(top_videos) > 5:
        print(f"  ... and {len(top_videos) - 5} more")
    
    return [v["url"] for v in top_videos]


def basic_discovery(query:str,max_results:int = 3):
    print(f"[discovery] Basic search: '{query}' (max {max_results} results)")
    
    search_url = f"ytsearch{max_results}:{query}"
    urls = []
    
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "skip_download": True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(search_url, download=False)
        entries = info.get("entries", [])
        for entry in entries:
            video_id = entry.get("id")
            if video_id:
                urls.append(f"https://www.youtube.com/watch?v={video_id}")
    
    print(f"[discovery] Found {len(urls)} URLs")
    return urls



            


def discover_from_list(urls:list[str]):
    valid = []
    for url in urls:
        url = normalize_url(url)
        if is_youtube_url(url):
            valid.append(url)
        else:
            print(f"[discovery] Skipping invalid URL: {url}")
    print(f"[discovery] {len(valid)} valid URLs found from list.")
    return valid

def discover_from_search(query:str, max_results:int=5):
    print(f"[discovery] Searching YouTube: '{query}' (max {max_results} results)")
    search_url = f"ytsearch{max_results}:{query}"
    urls = []

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "skip_download": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(search_url, download=False)
        entries = info.get("entries", [])
        for entry in entries:
            vid_id = entry.get("id")
            if vid_id:
                urls.append(f"https://www.youtube.com/watch?v={vid_id}")
    print(f"[discovery] Found {len(urls)} URLs for query: '{query}'")
    return urls

def discover_from_channel(channel_url:str, max_results:int=10):
    print(f"[discovery] Extracting URLs from channel: {channel_url}")

    ydl_opts = {
        "quiet":True,
        "no_warnings":True,
        "extract_flat":True,
        "playlistend": max_results
    }

    urls = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(channel_url,download=False)
        entries = info.get("entries", [])
        for entry in entries:
            vid_id = entry.get("id")
            if vid_id:
                urls.append(f"https://www.youtube.com/watch?v={vid_id}")
    print(f"[discovery] Found {len(urls)} URLs from channel.")
    return urls

def discover(
    urls: list[str] | None = None,
    query: str | None = None,
    channel_url: str | None = None,
    max_results: int = 5,
    smart_search: bool = True,
    min_duration: int  = 30,
    max_duration: int  = 3600,
    sort_by: str = "relevance",
):
    if urls:
        return discover_from_list(urls)
    elif query:
        if smart_search:
            return smart_discovery(
                query=query,
                max_results=max_results,
                min_duration=min_duration,
                max_duration=max_duration,
                sort_by=sort_by,)
        else:
            return basic_discovery(query,max_results)
    elif channel_url:
        return discover_from_channel(channel_url, max_results)
    else:
        raise ValueError("[discovery] Provide at least one of: urls, query, or channel_url.")