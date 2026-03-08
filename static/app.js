/**
 * NexusAI Router — Frontend Application
 *
 * Handles model selection, prompt submission, response rendering,
 * usage tracking, and system health monitoring.
 */

(function () {
    'use strict';

    // ── State ──────────────────────────────────────────────────────────
    const state = {
        models: [],
        selectedModelId: null,
        autoSelect: true,
        loading: false,
    };

    // ── DOM References ─────────────────────────────────────────────────
    const dom = {
        userIdInput: document.getElementById('user-id-input'),
        autoSelectToggle: document.getElementById('auto-select-toggle'),
        modelSelect: document.getElementById('model-select'),
        promptInput: document.getElementById('prompt-input'),
        charCount: document.getElementById('char-count'),
        sendBtn: document.getElementById('send-btn'),
        sendBtnText: document.querySelector('.send-btn-text'),
        sendBtnLoader: document.querySelector('.send-btn-loader'),
        responseArea: document.getElementById('response-area'),
        responseModelInfo: document.getElementById('response-model-info'),
        responseMeta: document.getElementById('response-meta'),
        responseBody: document.getElementById('response-body'),
        responseRecommendation: document.getElementById('response-recommendation'),
        errorArea: document.getElementById('error-area'),
        errorMessage: document.getElementById('error-message'),
        usageRequests: document.getElementById('usage-requests'),
        usageSystemTokens: document.getElementById('usage-system-tokens'),
        usageModelTokens: document.getElementById('usage-model-tokens'),
        usagePerModel: document.getElementById('usage-per-model'),
        healthDot: document.getElementById('health-dot'),
        healthBtn: document.getElementById('health-btn'),
        healthPanel: document.getElementById('health-panel'),
        healthModal: document.getElementById('health-modal'),
        healthModalClose: document.getElementById('health-modal-close'),
        healthModalBody: document.getElementById('health-modal-body'),
        toastContainer: document.getElementById('toast-container'),
    };

    // ── API Helpers ────────────────────────────────────────────────────
    const API = {
        async fetchModels() {
            const res = await fetch('/api/models');
            if (!res.ok) throw new Error('Failed to fetch models');
            return (await res.json()).models;
        },

        async chat(userId, prompt, modelId) {
            const body = { user_id: userId, prompt };
            if (modelId) body.model_id = modelId;

            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });

            if (!res.ok) {
                const data = await res.json().catch(() => ({}));
                throw new Error(data.detail || `Error ${res.status}`);
            }
            return res.json();
        },

        async fetchUsage(userId) {
            const res = await fetch(`/api/usage/${encodeURIComponent(userId)}`);
            if (!res.ok) throw new Error('Failed to fetch usage');
            return res.json();
        },

        async fetchHealth() {
            const res = await fetch('/api/health');
            if (!res.ok) throw new Error('Failed to fetch health');
            return res.json();
        },
    };

    // ── Toast Notifications ────────────────────────────────────────────
    function showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        dom.toastContainer.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(50px)';
            toast.style.transition = 'all 0.3s ease-out';
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }

    // ── Model Dropdown ────────────────────────────────────────────────
    function renderModelDropdown() {
        dom.modelSelect.innerHTML = '';
        state.models.forEach(model => {
            const opt = document.createElement('option');
            opt.value = model.id;
            opt.textContent = model.name;
            if (!model.available) {
                opt.disabled = true;
                opt.textContent += ' (unavailable)';
            }
            dom.modelSelect.appendChild(opt);
        });

        // Sync dropdown state with auto-select toggle
        dom.modelSelect.disabled = state.autoSelect;
        if (!state.autoSelect && state.selectedModelId) {
            dom.modelSelect.value = state.selectedModelId;
        }
    }

    // ── Response Rendering ─────────────────────────────────────────────
    function showResponse(data) {
        dom.responseArea.hidden = false;
        dom.errorArea.classList.remove('visible');

        dom.responseModelInfo.innerHTML = `
            <span class="response-model-name">${data.model_name}</span>
            ${data.auto_selected ? '<span class="response-model-badge">Auto-selected</span>' : ''}
        `;

        dom.responseMeta.innerHTML = `
            <span>⚡ ${data.latency_ms.toFixed(0)}ms</span>
            <span>📊 ${data.tokens_used} tokens</span>
            <span>💎 ${data.system_tokens_used} sys tokens</span>
        `;

        // Typing effect
        dom.responseBody.textContent = '';
        typeText(dom.responseBody, data.response_text, 8);

        if (data.recommendation) {
            dom.responseRecommendation.hidden = false;
            dom.responseRecommendation.textContent =
                `🧠 ${data.recommendation.reasoning} (confidence: ${(data.recommendation.confidence * 100).toFixed(0)}%)`;
        } else {
            dom.responseRecommendation.hidden = true;
        }
    }

    function typeText(element, text, delayMs) {
        let i = 0;
        const chunkSize = 3;
        function type() {
            if (i < text.length) {
                element.textContent += text.slice(i, i + chunkSize);
                i += chunkSize;
                setTimeout(type, delayMs);
            }
        }
        type();
    }

    function showError(message) {
        dom.errorArea.classList.add('visible');
        dom.responseArea.hidden = true;
        dom.errorMessage.textContent = message;
    }

    // ── Usage Rendering ────────────────────────────────────────────────
    async function refreshUsage() {
        try {
            const userId = dom.userIdInput.value.trim();
            if (!userId) return;
            const usage = await API.fetchUsage(userId);

            dom.usageRequests.textContent = usage.total_requests;
            dom.usageSystemTokens.textContent = usage.total_system_tokens.toLocaleString();
            dom.usageModelTokens.textContent = usage.total_model_tokens.toLocaleString();

            const perModel = usage.per_model;
            if (Object.keys(perModel).length === 0) {
                dom.usagePerModel.innerHTML = '<p class="no-data">No usage yet</p>';
            } else {
                dom.usagePerModel.innerHTML = Object.entries(perModel).map(([modelId, data]) => {
                    const model = state.models.find(m => m.id === modelId);
                    const name = model ? model.name : modelId;
                    return `
                        <div class="usage-model-item">
                            <div class="usage-model-name">${name}</div>
                            <div class="usage-model-stats">
                                <span>${data.request_count} reqs</span>
                                <span>${data.model_tokens} tokens</span>
                                <span>${data.system_tokens} sys</span>
                            </div>
                        </div>
                    `;
                }).join('');
            }
        } catch (e) {
            // Silently fail for usage refresh
        }
    }

    // ── Health Rendering ───────────────────────────────────────────────
    const STATE_LABELS = { closed: 'operational', open: 'blocked', half_open: 'degraded' };

    async function refreshHealth() {
        try {
            const health = await API.fetchHealth();
            const anyDown = health.models.some(m => !m.available);
            dom.healthDot.className = 'status-dot' + (anyDown ? ' degraded' : '');

            dom.healthPanel.innerHTML = health.models.map(m => {
                const label = STATE_LABELS[m.circuit_state] || m.circuit_state;
                return `
                <div class="health-item">
                    <span class="health-item-name">${m.model_name}</span>
                    <span class="health-item-state ${label}">
                        <span class="model-card-status ${m.available ? '' : 'unavailable'}"></span>
                        ${label}
                    </span>
                </div>
            `;
            }).join('');
        } catch (e) {
            dom.healthDot.className = 'status-dot unhealthy';
        }
    }

    function showHealthModal() {
        API.fetchHealth().then(health => {
            dom.healthModalBody.innerHTML = health.models.map(m => {
                const label = STATE_LABELS[m.circuit_state] || m.circuit_state;
                return `
                <div class="health-modal-model">
                    <h3>${m.model_name}</h3>
                    <div class="health-modal-row"><span>State</span><span class="health-item-state ${label}">${label}</span></div>
                    <div class="health-modal-row"><span>Available</span><span>${m.available ? '✅' : '❌'}</span></div>
                    <div class="health-modal-row"><span>Failure Rate</span><span>${(m.failure_rate * 100).toFixed(1)}%</span></div>
                    ${m.cooloff_remaining_seconds ? `<div class="health-modal-row"><span>Cooloff Remaining</span><span>${m.cooloff_remaining_seconds}s</span></div>` : ''}
                </div>
            `;
            }).join('') + `
                <div class="health-modal-model">
                    <h3>System</h3>
                    <div class="health-modal-row"><span>Status</span><span>${health.status}</span></div>
                    <div class="health-modal-row"><span>Total Requests</span><span>${health.total_requests_served}</span></div>
                    <div class="health-modal-row"><span>Uptime</span><span>${health.uptime_seconds.toFixed(0)}s</span></div>
                </div>
            `;
            dom.healthModal.classList.add('active');
        }).catch(() => showToast('Failed to load health data', 'error'));
    }

    // ── Submit Handler ─────────────────────────────────────────────────
    async function handleSubmit() {
        if (state.loading) return;

        const prompt = dom.promptInput.value.trim();
        const userId = dom.userIdInput.value.trim();

        if (!prompt) {
            showToast('Please enter a prompt', 'warning');
            return;
        }
        if (!userId) {
            showToast('Please enter a user ID', 'warning');
            return;
        }

        state.loading = true;
        dom.sendBtn.disabled = true;
        dom.sendBtnText.hidden = true;
        dom.sendBtnLoader.hidden = false;
        dom.errorArea.classList.remove('visible');
        dom.responseArea.hidden = true;

        try {
            const modelId = state.autoSelect ? null : state.selectedModelId;
            const data = await API.chat(userId, prompt, modelId);
            showResponse(data);
            showToast('Response received', 'success');
        } catch (e) {
            showError(e.message);
            showToast(e.message, 'error');
        } finally {
            state.loading = false;
            dom.sendBtn.disabled = false;
            dom.sendBtnText.hidden = false;
            dom.sendBtnLoader.hidden = true;
            refreshUsage();
            refreshHealth();
        }
    }

    // ── Event Listeners ────────────────────────────────────────────────
    dom.autoSelectToggle.addEventListener('change', (e) => {
        state.autoSelect = e.target.checked;
        dom.modelSelect.disabled = state.autoSelect;
        if (!state.autoSelect && state.models.length && !state.selectedModelId) {
            state.selectedModelId = state.models[0].id;
            dom.modelSelect.value = state.selectedModelId;
        }
    });

    dom.modelSelect.addEventListener('change', (e) => {
        state.selectedModelId = e.target.value;
    });

    dom.promptInput.addEventListener('input', () => {
        const len = dom.promptInput.value.length;
        dom.charCount.textContent = `${len.toLocaleString()} / 10,000`;
    });

    dom.sendBtn.addEventListener('click', handleSubmit);

    dom.promptInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
            handleSubmit();
        }
    });

    dom.healthBtn.addEventListener('click', showHealthModal);
    dom.healthModalClose.addEventListener('click', () => { dom.healthModal.classList.remove('active'); });
    dom.healthModal.addEventListener('click', (e) => {
        if (e.target === dom.healthModal) dom.healthModal.classList.remove('active');
    });

    // ── Initialization ─────────────────────────────────────────────────
    async function init() {
        try {
            state.models = await API.fetchModels();
            renderModelDropdown();
            await refreshHealth();
            await refreshUsage();
        } catch (e) {
            showToast('Failed to connect to server. Is it running?', 'error');
        }

        // Periodic health + usage refresh
        setInterval(refreshHealth, 10000);
        setInterval(refreshUsage, 15000);
    }

    init();
})();
