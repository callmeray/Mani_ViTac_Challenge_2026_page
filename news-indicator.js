const LATEST_NEWS_DATE = '2025-11-15';
const NEWS_LAST_SEEN_KEY = 'challenge_news_last_seen';

function updateNewsBadge() {
  const newsLink = document.querySelector('.nav-link[data-nav="news"]');
  if (!newsLink) return;

  const lastSeen = localStorage.getItem(NEWS_LAST_SEEN_KEY);
  const hasNew = !lastSeen || lastSeen < LATEST_NEWS_DATE;

  if (hasNew) {
    newsLink.classList.add('nav-link--has-news');
  } else {
    newsLink.classList.remove('nav-link--has-news');
  }
}

function markNewsAsSeen() {
  try {
    localStorage.setItem(NEWS_LAST_SEEN_KEY, LATEST_NEWS_DATE);
  } catch (e) {
  }
}

function setupNewsSeenHandlers() {
  const newsLink = document.querySelector('.nav-link[data-nav="news"]');
  if (newsLink) {
    newsLink.addEventListener('click', function () {
      markNewsAsSeen();
      updateNewsBadge();
    });
  }

  const newsSection = document.querySelector('#news');
  if (!newsSection) return;

  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          markNewsAsSeen();
          updateNewsBadge();
          observer.disconnect();
        }
      });
    }, { rootMargin: '0px 0px -40% 0px' });

    observer.observe(newsSection);
  } else {
    newsSection.addEventListener('focusin', function () {
      markNewsAsSeen();
      updateNewsBadge();
    });
  }
}

document.addEventListener('DOMContentLoaded', function () {
  updateNewsBadge();

  if (document.body.classList.contains('news-page')) {
    markNewsAsSeen();
  }

  setupNewsSeenHandlers();
});
