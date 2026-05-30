
// Initialize Lucide icons
lucide.createIcons();

// App State
const state = {
recommendations: [],
favorites: [],
currentMood: '',
isLoading: false,
lastSearchQuery: '',
currentUser: null,
authToken: localStorage.getItem('moodtune_token')
};

// Mood Word Banks for Random Generation
const moodWords = {
scenes: [
    'coffee shop', 'rainy day', 'sunny beach', 'forest walk', 'mountain top',
    'city lights', 'space station', 'cozy cabin', 'desert road', 'lake house',
    'winter night', 'spring garden', 'summer party', 'autumn park', 'tropical island',
    'bookstore', 'art gallery', 'train journey', 'campfire', 'starry night'
],
activities: [
    'morning', 'evening', 'night', 'afternoon', 'study session', 'workout',
    'road trip', 'yoga', 'meditation', 'coding', 'writing', 'painting',
    'cooking', 'cleaning', 'showering', 'commuting', 'gaming', 'dancing',
    'reading', 'dreaming', 'exploring', 'creating', 'reflecting', 'celebrating'
],
emotions: [
    'happy', 'sad', 'energetic', 'calm', 'focused', 'relaxed', 'nostalgic',
    'hopeful', 'romantic', 'melancholic', 'epic', 'mysterious', 'dreamy',
    'intense', 'peaceful', 'joyful', 'thoughtful', 'adventurous', 'cozy',
    'powerful', 'serene', 'lonely', 'triumphant', 'yearning'
],
genres: [
    'lofi', 'jazz', 'electronic', 'classical', 'rock', 'pop', 'ambient',
    'synthwave', 'folk', 'indie', 'r&b', 'hip hop', 'reggae', 'blues',
    'orchestral', 'chillhop', 'downtempo', 'house', 'techno', 'acoustic'
],
intensities: [
    'soft', 'gentle', 'moderate', 'intense', 'powerful', 'explosive',
    'mellow', 'subtle', 'strong', 'dynamic', 'building', 'crescendo'
],
times: [
    'sunrise', 'sunset', 'midnight', 'dawn', 'dusk', 'golden hour',
    'blue hour', 'afternoon', 'early morning', 'late night'
]
};

// Mood Color Mapping
const moodColors = {
'happy': {
    primary: '#FFD93D',
    secondary: '#FF9C3D',
    intensity: 85,
    emoji: '😊',
    particles: 8
},
'sad': {
    primary: '#6B73FF',
    secondary: '#000DFF',
    intensity: 30,
    emoji: '😢',
    particles: 3
},
'energetic': {
    primary: '#FF6B6B',
    secondary: '#FF8E53',
    intensity: 95,
    emoji: '⚡',
    particles: 12
},
'chill': {
    primary: '#4ECDC4',
    secondary: '#44A08D',
    intensity: 45,
    emoji: '😌',
    particles: 5
},
'focus': {
    primary: '#45B7D1',
    secondary: '#96C93D',
    intensity: 75,
    emoji: '🎯',
    particles: 6
},
'romantic': {
    primary: '#FF9A9E',
    secondary: '#FAD0C4',
    intensity: 65,
    emoji: '💖',
    particles: 7
},
'party': {
    primary: '#A8FF78',
    secondary: '#78FFD6',
    intensity: 90,
    emoji: '🎉',
    particles: 15
},
'angry': {
    primary: '#FF416C',
    secondary: '#FF4B2B',
    intensity: 80,
    emoji: '😠',
    particles: 10
},
'melancholic': {
    primary: '#7B4397',
    secondary: '#DC2430',
    intensity: 40,
    emoji: '🎭',
    particles: 4
},
'neutral': {
    primary: '#B8B8B8',
    secondary: '#7C7C7C',
    intensity: 50,
    emoji: '😐',
    particles: 2
},
'ecstatic': {
    primary: '#FFE000',
    secondary: '#FF0080',
    intensity: 100,
    emoji: '🤩',
    particles: 20
},
'depressed': {
    primary: '#667eea',
    secondary: '#764ba2',
    intensity: 20,
    emoji: '😔',
    particles: 2
},
'dance': {
    primary: '#FF0099',
    secondary: '#493240',
    intensity: 88,
    emoji: '💃',
    particles: 10
}
};

// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// DOM Elements
const elements = {
moodInput: document.getElementById('mood-input'),
searchBtn: document.getElementById('search-btn'),
resultsSection: document.getElementById('results-section'),
tracksContainer: document.getElementById('tracks-container'),
currentMood: document.getElementById('current-mood'),
loading: document.getElementById('loading'),
exampleChips: document.getElementById('example-chips'),
favoriteCount: document.getElementById('favorite-count'),
resultsCount: document.getElementById('results-count'),
totalDuration: document.getElementById('total-duration'),
avgMatch: document.getElementById('avg-match'),
errorMessage: document.getElementById('error-message'),
authButtons: document.getElementById('auth-buttons'),
userMenu: document.getElementById('user-menu'),
userDropdown: document.getElementById('user-dropdown'),
dropdownUsername: document.getElementById('dropdown-username'),
dropdownEmail: document.getElementById('dropdown-email')
};

// Initialize app
async function init() {
generateRandomExamples();
loadTheme();
testBackendConnection();

// Check if user is logged in
if (state.authToken) {
    await loadUserProfile();
} else {
    updateAuthUI();
}

// Add scroll effect to header
window.addEventListener('scroll', handleScroll);

// Add click listener to refresh examples
document.querySelector('.examples').addEventListener('click', function(e) {
    if (e.target.classList.contains('refresh-examples')) {
        generateRandomExamples();
    }
});

// Close dropdown when clicking outside
document.addEventListener('click', (e) => {
    if (!e.target.closest('#user-menu') && !e.target.closest('#user-dropdown')) {
        elements.userDropdown.style.display = 'none';
    }
});
}

// Scroll handler for header effect
function handleScroll() {
const header = document.querySelector('.header');
if (window.scrollY > 50) {
    header.classList.add('scrolled');
} else {
    header.classList.remove('scrolled');
}
}

// Test backend connection
async function testBackendConnection() {
try {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (response.ok) {
        const data = await response.json();
        console.log('✅ Backend connected successfully:', data);
    } else {
        console.warn('❌ Backend responded with error:', response.status);
    }
} catch (error) {
    console.warn('❌ Backend not reachable:', error);
    // Show user-friendly message
    setTimeout(() => {
        if (confirm('Backend server is not running. Would you like to see instructions on how to start it?')) {
            showBackendInstructions();
        }
    }, 1000);
}
}

function showBackendInstructions() {
alert('To start the backend server:\n\n1. Open terminal/command prompt\n2. Navigate to the project folder\n3. Run: python moodtune_backend.py\n4. Wait for "Starting FastAPI server..." message\n5. Then refresh this page');
}

// Authentication Functions
async function handleLogin(event) {
event.preventDefault();
const email = document.getElementById('login-email').value;
const password = document.getElementById('login-password').value;

try {
    const response = await fetch(`${API_BASE_URL}/auth/login`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, password })
    });

    const data = await response.json();

    if (response.ok) {
        state.authToken = data.access_token;
        localStorage.setItem('moodtune_token', data.access_token);
        await loadUserProfile();
        closeModal('login-modal');
        showFeedback('Successfully logged in!');
    } else {
        showError(data.detail || 'Login failed');
    }
} catch (error) {
    showError('Login failed: ' + error.message);
}
}

async function handleRegister(event) {
event.preventDefault();
const username = document.getElementById('register-username').value;
const email = document.getElementById('register-email').value;
const password = document.getElementById('register-password').value;
const confirmPassword = document.getElementById('register-confirm-password').value;

if (password !== confirmPassword) {
    showError('Passwords do not match');
    return;
}

try {
    const response = await fetch(`${API_BASE_URL}/auth/register`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, email, password })
    });

    const data = await response.json();

    if (response.ok) {
        showFeedback('Account created successfully! Please log in.');
        closeModal('register-modal');
        showLoginModal();
    } else {
        showError(data.detail || 'Registration failed');
    }
} catch (error) {
    showError('Registration failed: ' + error.message);
}
}

async function loadUserProfile() {
try {
    const response = await fetch(`${API_BASE_URL}/auth/me`, {
        headers: {
            'Authorization': `Bearer ${state.authToken}`
        }
    });

    if (response.ok) {
        const userData = await response.json();
        state.currentUser = userData;
        await loadUserFavorites();
        updateAuthUI();
    } else {
        // Token might be invalid
        logout();
    }
} catch (error) {
    console.error('Failed to load user profile:', error);
    logout();
}
}

async function loadUserFavorites() {
if (!state.authToken) return;

try {
    const response = await fetch(`${API_BASE_URL}/favorites`, {
        headers: {
            'Authorization': `Bearer ${state.authToken}`
        }
    });

    if (response.ok) {
        const favoritesData = await response.json();
        state.favorites = favoritesData.favorites || [];
        updateFavoriteCount();
    }
} catch (error) {
    console.error('Failed to load favorites:', error);
}
}

function logout() {
state.authToken = null;
state.currentUser = null;
state.favorites = [];
localStorage.removeItem('moodtune_token');
updateAuthUI();
elements.userDropdown.style.display = 'none';
showFeedback('Successfully logged out');

// Re-render tracks to update favorite hearts
if (state.recommendations.length > 0) {
    displayResults(state.recommendations, state.currentMood);
}
}

function updateAuthUI() {
if (state.currentUser) {
    elements.authButtons.style.display = 'none';
    elements.userMenu.style.display = 'flex';
    elements.dropdownUsername.textContent = state.currentUser.username;
    elements.dropdownEmail.textContent = state.currentUser.email;
} else {
    elements.authButtons.style.display = 'flex';
    elements.userMenu.style.display = 'none';
    elements.favoriteCount.textContent = '0';
}
}

function showUserMenu() {
const dropdown = elements.userDropdown;
dropdown.style.display = dropdown.style.display === 'none' ? 'block' : 'none';
}

function showLoginModal() {
closeModal('register-modal');
const modal = document.getElementById('login-modal');
modal.style.display = 'flex';
setupModalClose(modal, 'login-modal');
}

function showRegisterModal() {
closeModal('login-modal');
const modal = document.getElementById('register-modal');
modal.style.display = 'flex';
setupModalClose(modal, 'register-modal');
}

// Generate Random Examples
function generateRandomExamples() {
const numberOfExamples = 4;
elements.exampleChips.innerHTML = '';

for (let i = 0; i < numberOfExamples; i++) {
    const example = generateRandomMood();
    const chip = createExampleChip(example, i);
    elements.exampleChips.appendChild(chip);
}

// Add refresh button
const refreshChip = document.createElement('button');
refreshChip.className = 'example-chip refresh-examples';
refreshChip.innerHTML = '<i data-lucide="refresh-cw"></i> New Ideas';
refreshChip.title = 'Generate new random examples';
elements.exampleChips.appendChild(refreshChip);

lucide.createIcons();
}

function generateRandomMood() {
const templates = [
    // Scene + Activity
    () => `${randomWord(moodWords.scenes)} ${randomWord(moodWords.activities)}`,
    // Emotion + Activity
    () => `${randomWord(moodWords.emotions)} ${randomWord(moodWords.activities)}`,
    // Scene + Time
    () => `${randomWord(moodWords.scenes)} ${randomWord(moodWords.times)}`,
    // Emotion + Scene
    () => `${randomWord(moodWords.emotions)} ${randomWord(moodWords.scenes)}`,
    // Intensity + Emotion + Genre
    () => `${randomWord(moodWords.intensities)} ${randomWord(moodWords.emotions)} ${randomWord(moodWords.genres)}`,
    // Time + Activity + Scene
    () => `${randomWord(moodWords.times)} ${randomWord(moodWords.activities)} ${randomWord(moodWords.scenes)}`,
    // Genre + Scene + Emotion
    () => `${randomWord(moodWords.genres)} ${randomWord(moodWords.scenes)} ${randomWord(moodWords.emotions)}`,
    // Multiple emotions
    () => `${randomWord(moodWords.emotions)} and ${randomWord(moodWords.emotions)}`,
    // Complex scene description
    () => `${randomWord(moodWords.intensities)} ${randomWord(moodWords.scenes)} ${randomWord(moodWords.times)}`
];

const template = randomWord(templates);
return template();
}

function randomWord(wordArray) {
return wordArray[Math.floor(Math.random() * wordArray.length)];
}

function createExampleChip(example, index) {
const chip = document.createElement('button');
chip.className = 'example-chip';
chip.innerHTML = `<span class="example-emoji">${getRandomEmoji()}</span> ${example}`;

chip.addEventListener('click', () => {
    useExample(example);
    // Add a subtle animation when clicked
    chip.style.transform = 'scale(0.95)';
    setTimeout(() => chip.style.transform = 'scale(1)', 150);
});

// Staggered animation
chip.style.animationDelay = `${index * 0.1}s`;

return chip;
}

function getRandomEmoji() {
const emojis = ['🎵', '🎶', '🎧', '🎸', '🎹', '🥁', '🎷', '🎺', '🪕', '🎻', '✨', '🌟', '💫', '🔥', '💧', '🌊', '🍃', '🌙', '⭐', '⚡', '❤️', '🎉', '🌈', '🎨'];
return randomWord(emojis);
}

// Use example
function useExample(text) {
elements.moodInput.value = text;
elements.moodInput.focus();
// Auto-generate after a short delay to show the input change
setTimeout(() => getRecommendations(), 300);
}

// Theme Management
function loadTheme() {
const savedTheme = localStorage.getItem('moodtune_theme') || 'dark';
document.documentElement.setAttribute('data-theme', savedTheme);
updateThemeIcon(savedTheme);
}

function toggleTheme() {
const currentTheme = document.documentElement.getAttribute('data-theme');
const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

document.documentElement.setAttribute('data-theme', newTheme);
localStorage.setItem('moodtune_theme', newTheme);
updateThemeIcon(newTheme);
}

function updateThemeIcon(theme) {
const themeButton = document.querySelector('.btn-icon [data-lucide]');
if (themeButton) {
    const newIcon = theme === 'dark' ? 'sun' : 'moon';
    // Remove old icon and create new one
    themeButton.parentElement.innerHTML = `<i data-lucide="${newIcon}"></i>`;
    lucide.createIcons();
}
}

// Handle Enter key
function handleKeyPress(event) {
if (event.key === 'Enter') {
    getRecommendations();
}
}

// Get recommendations
async function getRecommendations() {
const moodText = elements.moodInput.value.trim();

if (!moodText) {
    showError('Please describe your mood');
    return;
}

// Save the search query for retry functionality
state.lastSearchQuery = moodText;

setLoading(true);
hideError();

try {
    console.log('Sending request to backend...');
    const response = await fetch(`${API_BASE_URL}/recommend`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        ...(state.authToken && { 'Authorization': `Bearer ${state.authToken}` })
        },
        body: JSON.stringify({
            text: moodText,
            limit: 8
        })
    });

    if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
    }

    const data = await response.json();
    console.log('Received data from backend:', data);

    if (data.personalized_recommendations && data.personalized_recommendations.length > 0) {
        displayResults(data.personalized_recommendations, moodText, data.top_5_songs_for_this_vibe);
    } else {
        throw new Error('No recommendations received');
    }

} catch (error) {
    console.error('Error:', error);
    showError(`Failed to get recommendations: ${error.message}\n\nMake sure the backend is running on ${API_BASE_URL}`);
} finally {
    setLoading(false);
}
}

// Set loading state
function setLoading(loading) {
state.isLoading = loading;
elements.searchBtn.disabled = loading;

if (loading) {
    elements.loading.style.display = 'block';
    elements.searchBtn.innerHTML = '<i data-lucide="loader" class="animate-spin"></i><span>Generating...</span>';
    elements.resultsSection.style.display = 'none';
} else {
    elements.loading.style.display = 'none';
    elements.searchBtn.innerHTML = '<i data-lucide="zap"></i><span>Generate</span>';
}
lucide.createIcons();
}

// Display results
function displayResults(recommendations, moodText, topSongs = []) {
state.recommendations = recommendations;
state.currentMood = moodText;

elements.currentMood.textContent = `"${moodText}"`;
elements.resultsCount.textContent = `${recommendations.length} tracks matching your vibe`;

// Add mood visualization
addMoodVisualization(moodText);

// Calculate playlist stats
const totalDuration = recommendations.reduce((sum, track) => sum + (track.duration || 180), 0);
const avgMatch = Math.round(recommendations.reduce((sum, track) => sum + (track.match_percentage || (track.similarity_score || 0.8) * 100), 0) / recommendations.length);

elements.totalDuration.textContent = `~${Math.round(totalDuration / 60)} min`;
elements.avgMatch.textContent = `${avgMatch}% avg match`;

// Clear previous results
elements.tracksContainer.innerHTML = '';

// Add top songs section if available
if (topSongs && topSongs.length > 0) {
    const topSongsSection = createTopSongsSection(topSongs);
    elements.tracksContainer.appendChild(topSongsSection);
}

// Add personalized recommendations with streaming links
recommendations.forEach((track, index) => {
    const trackElement = createTrackElement(track, index);
    elements.tracksContainer.appendChild(trackElement);
});

// Show results with animation
elements.resultsSection.style.display = 'block';
elements.resultsSection.classList.add('mood-transition-enter');
elements.resultsSection.scrollIntoView({ behavior: 'smooth' });


// Reinitialize icons for the new content
lucide.createIcons();
}

// Add mood visualization to results
function addMoodVisualization(moodText) {
const detectedMood = moodText.toLowerCase();
const moodConfig = getMoodConfig(detectedMood);

// Create or update mood visualization
let moodVisuals = document.getElementById('mood-visuals');
if (!moodVisuals) {
    moodVisuals = document.createElement('div');
    moodVisuals.id = 'mood-visuals';
    moodVisuals.className = 'mood-visuals';

    // Insert after mood badge
    const moodBadge = document.querySelector('.mood-badge');
    moodBadge.parentNode.insertBefore(moodVisuals, moodBadge.nextSibling);
}

moodVisuals.innerHTML = `
    <div class="mood-color-indicator mood-${moodConfig.name}" style="background: linear-gradient(135deg, ${moodConfig.primary}, ${moodConfig.secondary})"></div>
    <div class="mood-intensity-bar">
        <div class="mood-intensity-fill mood-${moodConfig.name}" style="width: ${moodConfig.intensity}%"></div>
    </div>
    <span class="mood-emoji">${moodConfig.emoji}</span>
    <span class="mood-label">${moodConfig.name.charAt(0).toUpperCase() + moodConfig.name.slice(1)} Vibe</span>
    <div class="mood-particles" id="mood-particles"></div>
`;

// Add particles
createMoodParticles(moodConfig);

// Enhance mood badge
const moodBadge = document.querySelector('.mood-badge');
moodBadge.classList.add('enhanced');
moodBadge.style.background = `linear-gradient(135deg, ${moodConfig.primary}, ${moodConfig.secondary})`;
}

// Get mood configuration
function getMoodConfig(moodText) {
// Try to match exact mood
for (const [mood, config] of Object.entries(moodColors)) {
    if (moodText.includes(mood)) {
        return { ...config, name: mood };
    }
}

// Fallback: analyze text for mood keywords
const moodKeywords = {
    'happy': ['happy', 'joy', 'good', 'great', 'amazing', 'wonderful'],
    'sad': ['sad', 'depressed', 'lonely', 'cry', 'heartbroken'],
    'energetic': ['energy', 'powerful', 'motivated', 'pumped', 'workout'],
    'chill': ['chill', 'relax', 'calm', 'peaceful', 'meditation'],
    'focus': ['focus', 'study', 'work', 'productive', 'concentrate']
};

for (const [mood, keywords] of Object.entries(moodKeywords)) {
    if (keywords.some(keyword => moodText.includes(keyword))) {
        return { ...moodColors[mood], name: mood };
    }
}

// Default to neutral
return { ...moodColors.neutral, name: 'neutral' };
}

// Create animated particles for mood
function createMoodParticles(moodConfig) {
const particlesContainer = document.getElementById('mood-particles');
particlesContainer.innerHTML = '';

for (let i = 0; i < moodConfig.particles; i++) {
    const particle = document.createElement('div');
    particle.className = 'particle';
    particle.style.cssText = `
        width: ${Math.random() * 8 + 4}px;
        height: ${Math.random() * 8 + 4}px;
        background: ${moodConfig.primary};
        top: ${Math.random() * 100}%;
        left: ${Math.random() * 100}%;
        animation-delay: ${Math.random() * 2}s;
        opacity: ${Math.random() * 0.4 + 0.2};
    `;
    particlesContainer.appendChild(particle);
}
}

// Create top songs section
function createTopSongsSection(topSongs) {
const section = document.createElement('div');
section.className = 'top-songs-section';
section.innerHTML = `
    <div class="section-header">
        <h4>🎯 Top Songs for This Vibe</h4>
        <p>Most popular tracks matching your mood</p>
    </div>
    <div class="top-songs-grid">
        ${topSongs.map(song => {
            const spotifyUrl = `https://open.spotify.com/search/${encodeURIComponent(song.song_name + ' ' + song.artist_name)}`;
            const appleMusicUrl = `https://music.apple.com/us/search?term=${encodeURIComponent(song.song_name + ' ' + song.artist_name)}`;

            return `
                <div class="top-song-card">
                    <div class="song-rank">${song.rank}</div>
                    <div class="song-info">
                        <div class="song-title">${escapeHtml(song.song_name)}</div>
                        <div class="song-artist">${escapeHtml(song.artist_name)}</div>
                    </div>
                    <div class="song-actions">
                        <a href="${spotifyUrl}" target="_blank" class="btn-action spotify-link" title="Search on Spotify">
                            <i data-lucide="music"></i>
                        </a>
                        <a href="${appleMusicUrl}" target="_blank" class="btn-action apple-music-link" title="Search on Apple Music">
                            <i data-lucide="play-circle"></i>
                        </a>
                    </div>
                </div>
            `;
        }).join('')}
    </div>
`;
return section;
}

// Create track element with streaming links - ALWAYS SHOW FAVORITE HEARTS
function createTrackElement(track, index) {
const artistName = track.artist_name || 'Unknown Artist';
const songName = track.song_name || 'Unknown Track';
const trackId = track.track_id || `track-${index}-${Date.now()}`;

const isFavorite = state.favorites.some(fav => fav.track_id === trackId);

// Generate search URLs for streaming services
const spotifyUrl = `https://open.spotify.com/search/${encodeURIComponent(songName + ' ' + artistName)}`;
const appleMusicUrl = `https://music.apple.com/us/search?term=${encodeURIComponent(songName + ' ' + artistName)}`;

const trackElement = document.createElement('div');
trackElement.className = 'track-card';
trackElement.innerHTML = `
    <div class="track-number">${index + 1}</div>
    <div class="track-info">
        <div class="track-title">${escapeHtml(songName)}</div>
        <div class="track-artist">${escapeHtml(artistName)}</div>
        ${track.why_it_matches ? `<div class="track-reason">${track.why_it_matches}</div>` : ''}
    </div>
    <div class="track-meta">
        <div class="similarity-score">
            ${track.match_percentage || Math.round((track.similarity_score || 0.8) * 100)}% match
        </div>
        <div class="track-actions">
            <a href="${spotifyUrl}" target="_blank" class="btn-action spotify-link" title="Search on Spotify">
                <i data-lucide="music"></i>
            </a>
            <a href="${appleMusicUrl}" target="_blank" class="btn-action apple-music-link" title="Search on Apple Music">
                <i data-lucide="play-circle"></i>
            </a>
            <button class="btn-action favorite ${isFavorite ? 'active' : ''}" onclick="handleFavoriteClick(${JSON.stringify({...track, track_id: trackId, artist_name: artistName, song_name: songName}).replace(/"/g, '&quot;')})">
                <i data-lucide="heart" ${isFavorite ? 'fill="currentColor"' : ''}></i>
            </button>
            <button class="btn-action" onclick="shareTrack(${JSON.stringify({...track, track_id: trackId, artist_name: artistName, song_name: songName}).replace(/"/g, '&quot;')})">
                <i data-lucide="share-2"></i>
            </button>
        </div>
    </div>
`;

return trackElement;
}

// Handle favorite click - show login modal if not authenticated
function handleFavoriteClick(track) {
if (!state.currentUser) {
    showFeedback('Please log in to save favorites');
    showLoginModal();
    return;
}
toggleFavorite(track);
}

// Toggle favorite
async function toggleFavorite(track) {
if (!state.currentUser) {
    showFeedback('Please log in to save favorites');
    showLoginModal();
    return;
}

const trackId = track.track_id;
const isCurrentlyFavorite = state.favorites.some(fav => fav.track_id === trackId);

try {
    const response = await fetch(`${API_BASE_URL}/favorites`, {
        method: isCurrentlyFavorite ? 'DELETE' : 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${state.authToken}`
        },
        body: JSON.stringify({
            track_id: trackId,
            song_name: track.song_name,
            artist_name: track.artist_name,
            album: track.album,
            match_percentage: track.match_percentage
        })
    });

    if (response.ok) {
        if (isCurrentlyFavorite) {
            state.favorites = state.favorites.filter(fav => fav.track_id !== trackId);
            showFeedback('Removed from favorites');
        } else {
            state.favorites.push(track);
            showFeedback('Added to favorites');
        }
        updateFavoriteCount();

        // Re-render the current results to update heart icons
        if (state.recommendations.length > 0) {
            displayResults(state.recommendations, state.currentMood);
        }
    } else {
        throw new Error('Failed to update favorite');
    }
} catch (error) {
    showError('Failed to update favorite: ' + error.message);
}
}

// Update favorite count
function updateFavoriteCount() {
elements.favoriteCount.textContent = state.favorites.length;
}

// Back to search
function backToSearch() {
elements.resultsSection.style.display = 'none';
elements.moodInput.focus();
}

// Clear results
function clearResults() {
elements.resultsSection.style.display = 'none';
elements.moodInput.value = '';
elements.moodInput.focus();
}

// Export playlist
function exportPlaylist() {
if (state.recommendations.length === 0) {
    showError('No tracks to export');
    return;
}

const playlist = {
    name: `MoodTune - ${state.currentMood}`,
    timestamp: new Date().toISOString(),
    tracks: state.recommendations
};

const dataStr = JSON.stringify(playlist, null, 2);
const dataBlob = new Blob([dataStr], { type: 'application/json' });

const link = document.createElement('a');
link.href = URL.createObjectURL(dataBlob);
link.download = `moodtune-${state.currentMood.replace(/[^a-z0-9]/gi, '-').toLowerCase()}.json`;
link.click();

showFeedback('Playlist exported!');
}

// Share playlist
function sharePlaylist() {
if (state.recommendations.length === 0) {
    showError('No tracks to share');
    return;
}

const playlistUrl = `${window.location.origin}${window.location.pathname}?mood=${encodeURIComponent(state.currentMood)}`;

if (navigator.share) {
    navigator.share({
        title: `MoodTune: ${state.currentMood}`,
        text: `Check out this playlist for "${state.currentMood}" created with MoodTune`,
        url: playlistUrl
    });
} else {
    navigator.clipboard.writeText(playlistUrl).then(() => {
        showFeedback('Playlist link copied to clipboard!');
    });
}
}

// Share track
function shareTrack(track) {
if (navigator.share) {
    navigator.share({
        title: track.song_name,
        text: `Check out "${track.song_name}" by ${track.artist_name} from MoodTune`,
        url: window.location.href
    });
} else {
    // Fallback: copy to clipboard
    const text = `${track.song_name} by ${track.artist_name}`;
    navigator.clipboard.writeText(text).then(() => {
        showFeedback('Track info copied to clipboard!');
    });
}
}

// Playlist Controls
function shufflePlaylist() {
const shuffled = [...state.recommendations];
for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
}
state.recommendations = shuffled;
displayResults(shuffled, state.currentMood);
showFeedback('Playlist shuffled');
}

// Error Handling
function showError(message) {
elements.errorMessage.textContent = message;
const errorModal = document.getElementById('error-modal');
errorModal.style.display = 'flex';

// Add escape key listener
const escapeHandler = (event) => {
    if (event.key === 'Escape') {
        hideError();
        document.removeEventListener('keydown', escapeHandler);
    }
};
document.addEventListener('keydown', escapeHandler);

// Also close on backdrop click
errorModal.onclick = (event) => {
    if (event.target === errorModal) {
        hideError();
    }
};
}

function hideError() {
const errorModal = document.getElementById('error-modal');
errorModal.style.display = 'none';
errorModal.onclick = null;
}

function retryLastSearch() {
if (state.lastSearchQuery) {
    elements.moodInput.value = state.lastSearchQuery;
    getRecommendations();
}
hideError();
}

// Favorites Modal
async function showFavorites() {
if (!state.currentUser) {
    showFeedback('Please log in to view favorites');
    showLoginModal();
    return;
}

const modal = document.getElementById('favorites-modal');
const favoritesList = document.getElementById('favorites-list');

if (state.favorites.length === 0) {
    favoritesList.innerHTML = '<div class="empty-state"><i data-lucide="heart" width="48" height="48"></i><p>No favorites yet</p><p class="empty-subtitle">Start exploring and add some tracks!</p></div>';
} else {
    favoritesList.innerHTML = state.favorites.map((track, index) => `
        <div class="track-card">
            <div class="track-number">${index + 1}</div>
            <div class="track-info">
                <div class="track-title">${escapeHtml(track.song_name)}</div>
                <div class="track-artist">${escapeHtml(track.artist_name)}</div>
            </div>
            <div class="track-actions">
                <a href="https://open.spotify.com/search/${encodeURIComponent(track.song_name + ' ' + track.artist_name)}" target="_blank" class="btn-action spotify-link" title="Search on Spotify">
                    <i data-lucide="music"></i>
                </a>
                <a href="https://music.apple.com/us/search?term=${encodeURIComponent(track.song_name + ' ' + track.artist_name)}" target="_blank" class="btn-action apple-music-link" title="Search on Apple Music">
                    <i data-lucide="play-circle"></i>
                </a>
                <button class="btn-action" onclick="toggleFavorite(${JSON.stringify(track).replace(/"/g, '&quot;')})">
                    <i data-lucide="x"></i>
                </button>
            </div>
        </div>
    `).join('');
}

modal.style.display = 'flex';
setupModalClose(modal, 'favorites-modal');

lucide.createIcons();
}

function showSettings() {
showFeedback('Settings would be shown here');
elements.userDropdown.style.display = 'none';
}

// Modal Functions
function setupModalClose(modal, modalId) {
modal.onclick = (event) => {
    if (event.target === modal) {
        closeModal(modalId);
    }
};

const escapeHandler = (event) => {
    if (event.key === 'Escape') {
        closeModal(modalId);
        document.removeEventListener('keydown', escapeHandler);
    }
};
document.addEventListener('keydown', escapeHandler);
}

function closeModal(modalId) {
const modal = document.getElementById(modalId);
modal.style.display = 'none';
modal.onclick = null;
}

function showAbout() {
const modal = document.getElementById('about-modal');
modal.style.display = 'flex';
setupModalClose(modal, 'about-modal');
}

function showPrivacy() {
showFeedback('Privacy policy would be shown here');
}

function showTerms() {
showFeedback('Terms of service would be shown here');
}

// Utility Functions
function escapeHtml(unsafe) {
if (!unsafe) return '';
return unsafe
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function showFeedback(message, type = 'success') {
// Create temporary feedback element
const feedback = document.createElement('div');
feedback.className = `feedback-toast feedback-${type}`;
feedback.innerHTML = `
    <div class="feedback-content">
        <i data-lucide="${type === 'success' ? 'check-circle' : 'alert-circle'}"></i>
        <span>${message}</span>
    </div>
`;

document.body.appendChild(feedback);

// Animate in
setTimeout(() => feedback.classList.add('show'), 10);

// Remove after delay
setTimeout(() => {
    feedback.classList.remove('show');
    setTimeout(() => {
        if (document.body.contains(feedback)) {
            document.body.removeChild(feedback);
        }
    }, 300);
}, 3000);

lucide.createIcons();
}

// Add feedback toast styles
const feedbackStyle = document.createElement('style');
feedbackStyle.textContent = `
.feedback-toast {
    position: fixed;
    top: 20px;
    right: 20px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    box-shadow: var(--shadow-lg);
    z-index: 3000;
    transform: translateX(100%);
    opacity: 0;
    transition: all 0.3s ease;
    max-width: 300px;
}

.feedback-toast.show {
    transform: translateX(0);
    opacity: 1;
}

.feedback-success {
    border-left: 4px solid var(--secondary);
}

.feedback-error {
    border-left: 4px solid var(--accent);
}

.feedback-content {
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.feedback-content i {
    width: 20px;
    height: 20px;
}

.feedback-success .feedback-content i {
    color: var(--secondary);
}

.feedback-error .feedback-content i {
    color: var(--accent);
}

.empty-state {
    text-align: center;
    padding: 3rem 2rem;
    color: var(--text-secondary);
}

.empty-state i {
    margin-bottom: 1rem;
    color: var(--text-muted);
}

.empty-subtitle {
    font-size: 0.9rem;
    margin-top: 0.5rem;
    opacity: 0.7;
}

.top-songs-section {
    margin-bottom: 2rem;
    padding: 1.5rem;
    background: var(--surface-light);
    border-radius: 16px;
    border: 1px solid var(--border-light);
}

.section-header {
    text-align: center;
    margin-bottom: 1.5rem;
}

.section-header h4 {
    color: var(--text-primary);
    margin-bottom: 0.5rem;
}

.section-header p {
    color: var(--text-secondary);
}

.top-songs-grid {
    display: grid;
    gap: 0.75rem;
}

.top-song-card {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1rem;
    background: var(--surface);
    border-radius: 12px;
    border: 1px solid var(--border);
    transition: all 0.3s ease;
}

.top-song-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow);
}

.song-rank {
    background: var(--gradient-primary);
    color: white;
    width: 36px;
    height: 36px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.9rem;
}

.song-info {
    flex: 1;
}

.song-title {
    font-weight: 600;
    color: var(--text-primary);
}

.song-artist {
    color: var(--text-secondary);
    font-size: 0.9rem;
}

.track-reason {
    margin-top: 0.5rem;
    font-size: 0.85rem;
    color: var(--text-secondary);
    font-style: italic;
}

/* Streaming service link styles */
.spotify-link {
    color: #1DB954 !important;
}

.apple-music-link {
    color: #FA243C !important;
}

.btn-action:hover.spotify-link {
    background: #1DB954 !important;
    color: white !important;
}

.btn-action:hover.apple-music-link {
    background: #FA243C !important;
    color: white !important;
}

/* Auth modal styles */
.auth-modal {
    max-width: 400px;
}

.form-group {
    margin-bottom: 1.5rem;
}

.form-group label {
    display: block;
    margin-bottom: 0.5rem;
    color: var(--text-primary);
    font-weight: 500;
}

.form-group input {
    width: 100%;
    padding: 0.75rem 1rem;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--surface);
    color: var(--text-primary);
    font-size: 1rem;
    transition: all 0.3s ease;
}

.form-group input:focus {
    outline: none;
    border-color: var(--primary);
    box-shadow: 0 0 0 3px rgba(245, 163, 163, 0.1);
}

.full-width {
    width: 100%;
}

.auth-footer {
    text-align: center;
    margin-top: 1.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border);
}

.auth-footer a {
    color: var(--primary);
    text-decoration: none;
    font-weight: 500;
}

.auth-footer a:hover {
    text-decoration: underline;
}

/* User dropdown styles */
.user-dropdown {
    position: absolute;
    top: 70px;
    right: 1.5rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    box-shadow: var(--shadow-lg);
    z-index: 1000;
    min-width: 200px;
    animation: dropdownSlide 0.2s ease;
}

@keyframes dropdownSlide {
    from {
        opacity: 0;
        transform: translateY(-10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.user-info {
    padding: 1rem;
    border-bottom: 1px solid var(--border);
}

.user-info span {
    display: block;
}

#dropdown-username {
    font-weight: 600;
    color: var(--text-primary);
}

.user-email {
    font-size: 0.85rem;
    color: var(--text-secondary);
    margin-top: 0.25rem;
}

.dropdown-divider {
    height: 1px;
    background: var(--border);
    margin: 0.5rem 0;
}

.dropdown-item {
    width: 100%;
    padding: 0.75rem 1rem;
    background: none;
    border: none;
    color: var(--text-primary);
    text-align: left;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    transition: background 0.3s ease;
}

.dropdown-item:hover {
    background: var(--surface-light);
}

.dropdown-item.logout {
    color: var(--accent);
}

.dropdown-item i {
    width: 16px;
    height: 16px;
}
`;
document.head.appendChild(feedbackStyle);

// Initialize when page loads
document.addEventListener('DOMContentLoaded', init);
