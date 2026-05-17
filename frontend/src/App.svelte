<script>
  import { onMount } from 'svelte';
  import CommentThread from './lib/CommentThread.svelte';
  import { api, getApiBaseUrl } from './lib/api.js';
  import { buildCommentTree, formatDate } from './lib/utils.js';

  const storageKey = 'arhis_token';

  let token = '';
  let me = null;
  let authMode = 'login';
  let activeTab = 'feed';

  let authForm = {
    username: '',
    password: ''
  };

  let postForm = {
    title: '',
    body: ''
  };

  let commentBody = '';
  let replyBody = '';

  let reportDraft = {
    reason: 'harassment',
    reasonText: ''
  };

  let decisionNotes = {};
  let reportsFilter = 'pending';
  let mlReportsFilter = 'all';

  let posts = [];
  let selectedPost = null;
  let comments = [];
  let commentTree = [];
  let reports = [];
  let mlReports = [];
  let users = [];
  let modelsOverview = null;
  let models = [];

  let replyTarget = null;
  let reportTarget = null;
  let userRoleDrafts = {};

  let authBusy = false;
  let loadingPosts = false;
  let loadingComments = false;
  let loadingReports = false;
  let loadingUsers = false;
  let loadingModels = false;
  let creatingPost = false;
  let creatingComment = false;
  let creatingReport = false;
  let updatingUserRole = '';
  let decidingReportId = '';
  let promotingModelVersion = '';
  let rollingBackModel = false;

  let authError = '';
  let appError = '';
  let notice = '';
  let feedRefreshHandle = null;
  let feedRefreshInProgress = false;

  const feedRefreshIntervalMs = 1000;

  $: canModerate = Boolean(me && (me.role === 'moderator' || me.role === 'admin'));
  $: canAdmin = Boolean(me && me.role === 'admin');
  $: currentUserName = me ? (me.display_name || me.username) : '';

  onMount(async () => {
    token = localStorage.getItem(storageKey) || '';
    if (token) {
      await restoreSession();
    }

    feedRefreshHandle = window.setInterval(() => {
      void refreshOpenPost();
    }, feedRefreshIntervalMs);

    return () => {
      if (feedRefreshHandle) {
        window.clearInterval(feedRefreshHandle);
        feedRefreshHandle = null;
      }
    };
  });

  function setNotice(message) {
    notice = message;
    appError = '';
  }

  function setError(message) {
    appError = message;
    notice = '';
  }

  function clearSession() {
    token = '';
    me = null;
    posts = [];
    selectedPost = null;
    comments = [];
    commentTree = [];
    reports = [];
    mlReports = [];
    users = [];
    modelsOverview = null;
    models = [];
    reportsFilter = 'pending';
    mlReportsFilter = 'all';
    replyTarget = null;
    reportTarget = null;
    userRoleDrafts = {};
    commentBody = '';
    replyBody = '';
    localStorage.removeItem(storageKey);
  }

  function isMlManagedReport(report) {
    return (
      report.status === 'queued_for_ml' ||
      report.status === 'under_review' ||
      report.decision_source === 'ml_auto' ||
      (report.status === 'resolved' && report.ml_score !== null && report.ml_score !== undefined)
    );
  }

  function getVerdictActorLabel(report) {
    if (report.decision_source === 'ml_auto') {
      return 'ML';
    }
    if (report.decision_source === 'manual') {
      return 'Moderator';
    }
    if (report.status === 'queued_for_ml') {
      return 'Pending ML';
    }
    if (report.status === 'under_review') {
      return 'ML escalated';
    }
    return 'Unknown';
  }

  function getConfidenceLabel(report) {
    if (report.ml_score === null || report.ml_score === undefined) {
      return 'n/a';
    }
    return Number(report.ml_score).toFixed(3);
  }

  function getMlModelStageLabel(report) {
    if (!report.ml_model_stage) {
      return 'n/a';
    }
    return report.ml_model_stage;
  }

  function getMlModelStageChipClass(report) {
    if (report.ml_model_stage === 'production') {
      return 'chip-prod';
    }
    if (report.ml_model_stage === 'canary') {
      return 'chip-canary';
    }
    return 'chip-soft';
  }

  async function restoreSession() {
    try {
      me = await api.me(token);
      await loadPosts({ silent: true });
      if (me && (me.role === 'moderator' || me.role === 'admin')) {
        await loadReports({ silent: true });
        await loadMlReports({ silent: true });
      }
      if (me && me.role === 'admin') {
        await loadUsers({ silent: true });
        await loadModels({ silent: true });
      }
    } catch (error) {
      clearSession();
      setError(error.message || 'Session expired. Please sign in again.');
    }
  }

  async function refreshOpenPost() {
    if (!token || feedRefreshInProgress || activeTab !== 'feed' || !selectedPost) {
      return;
    }

    if (
      authBusy ||
      creatingPost ||
      creatingComment ||
      creatingReport ||
      updatingUserRole ||
      promotingModelVersion ||
      rollingBackModel ||
      decidingReportId
    ) {
      return;
    }

    feedRefreshInProgress = true;

    try {
      if (activeTab === 'feed') {
        await refreshSelectedPost({ silent: true, preserveDrafts: true });
      }
    } catch (error) {
      console.debug('Auto refresh failed', error);
    } finally {
      feedRefreshInProgress = false;
    }
  }

  async function submitAuth() {
    authBusy = true;
    authError = '';

    try {
      const payload = {
        username: authForm.username.trim(),
        password: authForm.password
      };

      const response = authMode === 'login'
        ? await api.login(payload)
        : await api.register(payload);

      token = response.access_token;
      me = response.user;
      localStorage.setItem(storageKey, token);

      authForm = { username: '', password: '' };
      activeTab = 'feed';
      setNotice(authMode === 'login' ? 'Signed in.' : 'Account created.');
      await loadPosts({ silent: true });
      if (me && (me.role === 'moderator' || me.role === 'admin')) {
        await loadReports({ silent: true });
        await loadMlReports({ silent: true });
      }
      if (me && me.role === 'admin') {
        await loadUsers({ silent: true });
        await loadModels({ silent: true });
      }
    } catch (error) {
      authError = error.message || 'Authentication failed.';
    } finally {
      authBusy = false;
    }
  }

  async function logout() {
    clearSession();
    activeTab = 'feed';
    setNotice('Signed out.');
  }

  async function loadPosts({ silent = false } = {}) {
    loadingPosts = true;
    if (!silent) {
      appError = '';
    }

    try {
      const data = await api.listPosts();
      posts = Array.isArray(data) ? data : [];

      if (selectedPost) {
        const updated = posts.find((post) => post.id === selectedPost.id);
        if (updated) {
          selectedPost = updated;
        } else if (posts.length > 0) {
          await selectPost(posts[0]);
          return;
        } else {
          selectedPost = null;
          comments = [];
          commentTree = [];
        }
      }

      if (!selectedPost && posts.length) {
        await selectPost(posts[0]);
      }
    } catch (error) {
      setError(error.message || 'Failed to load posts.');
    } finally {
      loadingPosts = false;
    }
  }

  async function fetchPostView(postId) {
    const freshPost = await api.getPost(postId);
    const data = await api.listComments(postId);
    const nextComments = Array.isArray(data) ? data : [];

    return {
      freshPost,
      comments: nextComments,
      commentTree: buildCommentTree(nextComments)
    };
  }

  async function selectPost(post, { silent = false, preserveDrafts = false } = {}) {
    const draftState = preserveDrafts
      ? {
          replyTarget,
          reportTarget,
          commentBody,
          replyBody
        }
      : null;

    selectedPost = post;
    replyTarget = null;
    reportTarget = null;
    commentTree = [];
    commentBody = '';
    replyBody = '';
    loadingComments = true;

    try {
      if (!silent) {
        appError = '';
      }

      const nextState = await fetchPostView(post.id);
      selectedPost = nextState.freshPost;
      comments = nextState.comments;
      commentTree = nextState.commentTree;

      if (draftState) {
        replyTarget = draftState.replyTarget;
        reportTarget = draftState.reportTarget;
        commentBody = draftState.commentBody;
        replyBody = draftState.replyBody;
      }
    } catch (error) {
      if (!silent) {
        setError(error.message || 'Failed to load comments.');
      }
    } finally {
      loadingComments = false;
    }
  }

  async function createPost() {
    creatingPost = true;
    try {
      const payload = {
        title: postForm.title.trim(),
        body: postForm.body.trim()
      };

      const created = await api.createPost(token, payload);
      postForm = { title: '', body: '' };
      posts = [created, ...posts.filter((post) => post.id !== created.id)];
      await selectPost(created);
      activeTab = 'feed';
      setNotice('Post published.');
    } catch (error) {
      setError(error.message || 'Failed to create post.');
    } finally {
      creatingPost = false;
    }
  }

  async function submitComment() {
    if (!selectedPost) {
      setError('Select a post first.');
      return;
    }

    const body = commentBody.trim();
    if (!body) {
      setError('Comment cannot be empty.');
      return;
    }

    creatingComment = true;
    try {
      await api.createComment(token, selectedPost.id, {
        body,
        parent_comment_id: replyTarget ? replyTarget.id : null
      });

      commentFormClear();
      await refreshSelectedPost();
      setNotice('Comment added.');
    } catch (error) {
      setError(error.message || 'Failed to create comment.');
    } finally {
      creatingComment = false;
    }
  }

  function commentFormClear() {
    commentBody = '';
    replyTarget = null;
  }

  async function deleteComment(comment) {
    if (!confirm('Delete this comment?')) {
      return;
    }

    try {
      await api.deleteComment(token, comment.id);
      await refreshSelectedPost();
      setNotice('Comment deleted.');
    } catch (error) {
      setError(error.message || 'Failed to delete comment.');
    }
  }

  async function refreshSelectedPost({ silent = false, preserveDrafts = false } = {}) {
    const targetPost = selectedPost;
    if (!targetPost) {
      return;
    }

    const draftState = preserveDrafts
      ? {
          replyTarget,
          reportTarget,
          commentBody,
          replyBody
        }
      : null;

    try {
      if (!silent) {
        appError = '';
      }

      const nextState = await fetchPostView(targetPost.id);
      selectedPost = nextState.freshPost;
      comments = nextState.comments;
      commentTree = nextState.commentTree;

      if (draftState) {
        replyTarget = draftState.replyTarget;
        reportTarget = draftState.reportTarget;
        commentBody = draftState.commentBody;
        replyBody = draftState.replyBody;
      }
    } catch (error) {
      if (!silent) {
        setError(error.message || 'Failed to load comments.');
      }
    }
  }

  function startReply(comment) {
    replyTarget = comment;
    reportTarget = null;
    replyBody = '';
  }

  function startReport(comment) {
    reportTarget = comment;
    replyTarget = null;
    replyBody = '';
  }

  function closeReportModal() {
    reportTarget = null;
    reportDraft = { reason: 'harassment', reasonText: '' };
  }

  function closeReplyModal() {
    replyTarget = null;
    replyBody = '';
  }

  async function submitReply() {
    if (!selectedPost || !replyTarget) {
      return;
    }

    const body = replyBody.trim();
    if (!body) {
      setError('Reply cannot be empty.');
      return;
    }

    creatingComment = true;
    try {
      await api.createComment(token, selectedPost.id, {
        body,
        parent_comment_id: replyTarget.id
      });

      closeReplyModal();
      await refreshSelectedPost();
      setNotice('Reply added.');
    } catch (error) {
      setError(error.message || 'Failed to create reply.');
    } finally {
      creatingComment = false;
    }
  }

  async function submitReport() {
    if (!reportTarget) {
      return;
    }

    creatingReport = true;
    try {
      await api.reportComment(token, reportTarget.id, {
        reason: reportDraft.reason,
        reason_text: reportDraft.reasonText.trim() || null
      });
      closeReportModal();
      await loadReports({ silent: true });
      await loadMlReports({ silent: true });
      setNotice('Report sent.');
    } catch (error) {
      setError(error.message || 'Failed to submit report.');
    } finally {
      creatingReport = false;
    }
  }

  async function loadReports({ silent = false } = {}) {
    if (!canModerate) {
      reports = [];
      return;
    }

    loadingReports = true;
    if (!silent) {
      appError = '';
    }

    try {
      const status = reportsFilter === 'all' ? null : reportsFilter;
      const data = await api.listReports(token, status, { limit: 200 });
      reports = Array.isArray(data) ? data : [];
    } catch (error) {
      setError(error.message || 'Failed to load reports.');
    } finally {
      loadingReports = false;
    }
  }

  async function loadMlReports({ silent = false } = {}) {
    if (!canModerate) {
      mlReports = [];
      return;
    }

    loadingReports = true;
    if (!silent) {
      appError = '';
    }

    try {
      const data = await api.listReports(token, null, { limit: 200 });
      const fetchedReports = Array.isArray(data) ? data.filter(isMlManagedReport) : [];
      mlReports = fetchedReports.filter((report) => {
        if (mlReportsFilter === 'all') {
          return true;
        }
        return report.status === mlReportsFilter;
      });
    } catch (error) {
      setError(error.message || 'Failed to load ML reports.');
    } finally {
      loadingReports = false;
    }
  }

  async function loadUsers({ silent = false } = {}) {
    if (!canAdmin) {
      users = [];
      return;
    }

    loadingUsers = true;
    if (!silent) {
      appError = '';
    }

    try {
      const data = await api.listUsers(token);
      users = Array.isArray(data) ? data : [];
      userRoleDrafts = Object.fromEntries(users.map((user) => [user.id, user.role]));
    } catch (error) {
      setError(error.message || 'Failed to load users.');
    } finally {
      loadingUsers = false;
    }
  }

  async function loadModels({ silent = false } = {}) {
    if (!canAdmin) {
      modelsOverview = null;
      models = [];
      return;
    }

    loadingModels = true;
    if (!silent) {
      appError = '';
    }

    try {
      const data = await api.listModels(token);
      modelsOverview = data;
      models = Array.isArray(data?.versions) ? data.versions : [];
    } catch (error) {
      setError(error.message || 'Failed to load model registry.');
    } finally {
      loadingModels = false;
    }
  }

  function setUserRoleDraft(userId, role) {
    userRoleDrafts = {
      ...userRoleDrafts,
      [userId]: role
    };
  }

  async function saveUserRole(user) {
    const nextRole = userRoleDrafts[user.id] || user.role;
    if (nextRole === user.role) {
      return;
    }

    updatingUserRole = user.id;
    try {
      await api.updateUserRole(token, user.id, nextRole);
      await loadUsers({ silent: true });
      setNotice(`Role updated for ${user.username}.`);
    } catch (error) {
      setError(error.message || 'Failed to update user role.');
    } finally {
      updatingUserRole = '';
    }
  }

  async function promoteModel(version) {
    promotingModelVersion = version;
    try {
      await api.promoteModel(token, version);
      await loadModels({ silent: true });
      setNotice(`Model ${version} promoted to production.`);
    } catch (error) {
      setError(error.message || 'Failed to promote model.');
    } finally {
      promotingModelVersion = '';
    }
  }

  async function rollbackModel() {
    if (!confirm('Rollback the current production model to the previous version?')) {
      return;
    }

    rollingBackModel = true;
    try {
      await api.rollbackModel(token);
      await loadModels({ silent: true });
      setNotice('Production model rolled back.');
    } catch (error) {
      setError(error.message || 'Failed to rollback model.');
    } finally {
      rollingBackModel = false;
    }
  }

  async function setReportsFilter(nextFilter) {
    reportsFilter = nextFilter;
    await loadReports();
  }

  async function setMlReportsFilter(nextFilter) {
    mlReportsFilter = nextFilter;
    await loadMlReports();
  }

  async function decideReport(reportId, verdict) {
    decidingReportId = reportId;
    try {
      await api.decideReport(token, reportId, {
        verdict,
        note: decisionNotes[reportId] || ''
      });

      decisionNotes = { ...decisionNotes, [reportId]: '' };
      await loadReports({ silent: true });
      await loadMlReports({ silent: true });
      if (selectedPost) {
        await refreshSelectedPost();
      }
      setNotice(`Report marked as ${verdict}.`);
    } catch (error) {
      setError(error.message || 'Failed to save moderation decision.');
    } finally {
      decidingReportId = '';
    }
  }

  function switchTab(tab) {
    activeTab = tab;
    if (tab === 'moderation' && canModerate) {
      loadReports();
    }
    if (tab === 'ml-reports' && canModerate) {
      loadMlReports();
    }
    if (tab === 'users' && canAdmin) {
      loadUsers();
    }
    if (tab === 'ml' && canAdmin) {
      loadModels();
    }
  }

  $: authTitle = authMode === 'login' ? 'Welcome back' : 'Create your account';
  $: authButtonLabel = authMode === 'login' ? 'Sign in' : 'Register';
  $: userRole = me ? me.role : '';
</script>

<svelte:head>
  <title>Arhis</title>
  <meta
    name="description"
    content="Minimal Svelte frontend for the Arhis Reddit-like moderation app."
  />
</svelte:head>

{#if !token}
  <main class="auth-shell">
    <section class="card auth-card">
      <div class="eyebrow">Arhis</div>
      <h1 class="auth-title">Sign in or register</h1>
      <p class="muted">A minimal interface for reading, posting, and moderating discussions.</p>

      <div class="segmented">
        <button type="button" class:active={authMode === 'login'} on:click={() => (authMode = 'login')}>
          Sign in
        </button>
        <button type="button" class:active={authMode === 'register'} on:click={() => (authMode = 'register')}>
          Register
        </button>
      </div>

      <h2>{authTitle}</h2>
      <p class="muted">API: {getApiBaseUrl()}</p>

      <form class="stack" on:submit|preventDefault={submitAuth}>
        <label>
          <span>Username</span>
          <input bind:value={authForm.username} autocomplete="username" minlength="3" maxlength="64" required />
        </label>
        <label>
          <span>Password</span>
          <input
            bind:value={authForm.password}
            type="password"
            autocomplete={authMode === 'login' ? 'current-password' : 'new-password'}
            minlength="8"
            maxlength="128"
            required
          />
        </label>

        {#if authError}
          <div class="error-box">{authError}</div>
        {/if}

        <button class="primary-button" type="submit" disabled={authBusy}>
          {authBusy ? 'Please wait...' : authButtonLabel}
        </button>
      </form>
    </section>
  </main>
{:else}
  <main class="app-shell">
    <header class="topbar">
      <div>
        <div class="eyebrow">Arhis</div>
        <h1>Discussion board</h1>
      </div>

      <div class="topbar-actions">
        <div class="user-pill">
          <strong>{currentUserName}</strong>
          <span>{userRole}</span>
        </div>
        <button type="button" class="ghost-button" on:click={logout}>Logout</button>
      </div>
    </header>

    <nav class="tabs">
      <button type="button" class:active={activeTab === 'feed'} on:click={() => switchTab('feed')}>Feed</button>
      <button type="button" class:active={activeTab === 'create'} on:click={() => switchTab('create')}>Create</button>
      {#if canModerate}
        <button type="button" class:active={activeTab === 'moderation'} on:click={() => switchTab('moderation')}>
          Moderation
        </button>
        <button type="button" class:active={activeTab === 'ml-reports'} on:click={() => switchTab('ml-reports')}>
          ML reports
        </button>
      {/if}
      {#if canAdmin}
        <button type="button" class:active={activeTab === 'users'} on:click={() => switchTab('users')}>
          Users
        </button>
        <button type="button" class:active={activeTab === 'ml'} on:click={() => switchTab('ml')}>
          ML
        </button>
      {/if}
      <button type="button" class:active={activeTab === 'profile'} on:click={() => switchTab('profile')}>Profile</button>
    </nav>

    {#if notice}
      <div class="notice-box">{notice}</div>
    {/if}

    {#if appError}
      <div class="error-box">{appError}</div>
    {/if}

    {#if activeTab === 'feed'}
      <section class="workspace">
        <aside class="card panel">
          <div class="panel-header">
            <h2>Posts</h2>
          <button type="button" class="ghost-button" on:click={() => loadPosts()}>Refresh</button>
          </div>

          {#if loadingPosts}
            <div class="muted">Loading posts...</div>
          {:else if posts.length === 0}
            <div class="empty-state">
              <strong>No posts yet</strong>
              <p>Create the first post from the Create tab.</p>
            </div>
          {:else}
            <div class="post-list">
              {#each posts as post (post.id)}
                <button
                  type="button"
                  class:selected={selectedPost && selectedPost.id === post.id}
                  class="post-card"
                  on:click={() => selectPost(post)}
                >
                  <div class="post-card-header">
                    <h3>{post.title}</h3>
                    <span class="chip chip-soft">{post.comments_count} comments</span>
                  </div>
                  <p>{post.body}</p>
                  <div class="post-card-meta">
                    <span>{post.author_name}</span>
                    <span>{formatDate(post.created_at)}</span>
                  </div>
                </button>
              {/each}
            </div>
          {/if}
        </aside>

        <section class="card panel">
          {#if selectedPost}
            <section class="post-overview-card">
              <div class="panel-header">
                <div>
                  <h2>{selectedPost.title}</h2>
                  <p class="muted">
                    {selectedPost.author_name} · {formatDate(selectedPost.created_at)} · {selectedPost.comments_count} comments
                  </p>
                </div>
              </div>

              <article class="post-detail">
                <p>{selectedPost.body}</p>
              </article>
            </section>

            <div class="composer compact">
              <div class="panel-header">
                <h3>Write a comment</h3>
              </div>

              <textarea bind:value={commentBody} rows="3" placeholder="Share your thought..."></textarea>
              <div class="composer-actions">
                <button type="button" class="primary-button" on:click={submitComment} disabled={creatingComment}>
                  {creatingComment ? 'Posting...' : 'Post comment'}
                </button>
              </div>
            </div>

            <section class="comments-card">
              <div class="panel-header">
                <h3>Comments</h3>
                <button type="button" class="ghost-button" on:click={() => selectPost(selectedPost)}>Reload</button>
              </div>

              {#if loadingComments}
                <div class="muted">Loading comments...</div>
              {:else if commentTree.length === 0}
                <div class="empty-state">
                  <strong>No comments yet</strong>
                  <p>Start the thread with the form above.</p>
                </div>
              {:else}
                <CommentThread
                  comments={commentTree}
                  currentUserId={me.id}
                  onReply={startReply}
                  onReport={startReport}
                  onDelete={deleteComment}
                />
              {/if}
            </section>
          {:else}
            <div class="empty-state tall">
              <strong>Select a post</strong>
              <p>Pick a post from the list to read comments and reply.</p>
            </div>
          {/if}
        </section>
      </section>
    {:else if activeTab === 'create'}
      <section class="card panel narrow">
        <div class="panel-header">
          <h2>Create post</h2>
          <button type="button" class="ghost-button" on:click={() => switchTab('feed')}>Back to feed</button>
        </div>

        <form class="stack" on:submit|preventDefault={createPost}>
          <label>
            <span>Title</span>
            <input bind:value={postForm.title} minlength="1" maxlength="255" required />
          </label>
          <label>
            <span>Body</span>
            <textarea bind:value={postForm.body} rows="8" minlength="1" required></textarea>
          </label>
          <button class="primary-button" type="submit" disabled={creatingPost}>
            {creatingPost ? 'Publishing...' : 'Publish post'}
          </button>
        </form>
      </section>
    {:else if activeTab === 'moderation' && canModerate}
      <section class="card panel">
        <div class="panel-header">
          <h2>Moderation queue</h2>
          <div class="segmented compact">
            <button type="button" class:active={reportsFilter === 'pending'} on:click={() => setReportsFilter('pending')}>
              Pending
            </button>
            <button
              type="button"
              class:active={reportsFilter === 'under_review'}
              on:click={() => setReportsFilter('under_review')}
            >
              Under review
            </button>
            <button type="button" class:active={reportsFilter === 'all'} on:click={() => setReportsFilter('all')}>
              All
            </button>
          </div>
        </div>

        {#if loadingReports}
          <div class="muted">Loading reports...</div>
        {:else if reports.length === 0}
          <div class="empty-state">
            <strong>No reports</strong>
            <p>
              {#if reportsFilter === 'pending'}
                The pending queue is empty right now.
              {:else if reportsFilter === 'under_review'}
                No reports are currently waiting for manual review after ML.
              {:else}
                There are no reports for this view right now.
              {/if}
            </p>
          </div>
        {:else}
          <div class="report-list">
            {#each reports as report (report.id)}
              <article class="report-card">
                <div class="report-card-header">
                  <div>
                    <h3>{report.reason}</h3>
                    <p class="muted">
                      Status: {report.status} · verdict by {getVerdictActorLabel(report)} · comment author {report.comment_author_name} · reporter {report.reporter_name}
                    </p>
                  </div>
                  {#if report.decision_source}
                    <span class={`chip ${report.decision_source === 'ml_auto' ? 'chip-canary' : 'chip-soft'}`}>
                      {report.decision_source === 'ml_auto' ? 'ML verdict' : 'Moderator verdict'}
                    </span>
                  {:else if report.status === 'queued_for_ml'}
                    <span class="chip chip-canary">Queued for ML</span>
                  {/if}
                </div>

                <div class="report-meta-grid">
                  <div>
                    <span class="muted">ML confidence</span>
                    <strong>{getConfidenceLabel(report)}</strong>
                  </div>
                  <div>
                    <span class="muted">ML verdict</span>
                    <strong>{report.ml_verdict || 'n/a'}</strong>
                  </div>
                  <div>
                    <span class="muted">Model version</span>
                    <strong>{report.ml_model_version || 'n/a'}</strong>
                  </div>
                  <div>
                    <span class="muted">Model stage</span>
                    <strong>
                      {#if report.ml_model_stage}
                        <span class={`chip ${getMlModelStageChipClass(report)}`}>{getMlModelStageLabel(report)}</span>
                      {:else}
                        n/a
                      {/if}
                    </strong>
                  </div>
                  <div>
                    <span class="muted">Scored at</span>
                    <strong>{report.ml_scored_at ? formatDate(report.ml_scored_at) : 'n/a'}</strong>
                  </div>
                  <div>
                    <span class="muted">Decision by</span>
                    <strong>{getVerdictActorLabel(report)}</strong>
                  </div>
                </div>

                {#if report.decision_source === 'ml_auto'}
                  <div class="ml-decision-banner">
                    ML auto decision based on confidence and routing policy.
                  </div>
                {:else if report.status === 'queued_for_ml'}
                  <div class="ml-decision-banner muted">
                    Waiting for ML worker to score this report.
                  </div>
                {/if}

                <div class="reported-comment">
                  <span class="muted tiny">Reported comment</span>
                  <p>{report.comment_body}</p>
                </div>

                {#if report.reason_text}
                  <p>{report.reason_text}</p>
                {/if}

                <label>
                  <span>Moderator note</span>
                  <textarea
                    rows="3"
                  value={decisionNotes[report.id] || ''}
                  on:input={(event) => {
                    decisionNotes = {
                      ...decisionNotes,
                      [report.id]: event.currentTarget.value
                    };
                  }}
                  placeholder="Optional note for audit trail"
                ></textarea>
                </label>

                <div class="decision-actions">
                  <button
                    type="button"
                    class="primary-button"
                    on:click={() => decideReport(report.id, 'toxic')}
                    disabled={decidingReportId === report.id}
                  >
                    Toxic
                  </button>
                  <button
                    type="button"
                    class="ghost-button"
                    on:click={() => decideReport(report.id, 'not_toxic')}
                    disabled={decidingReportId === report.id}
                  >
                    Not toxic
                  </button>
                </div>
              </article>
            {/each}
          </div>
        {/if}
      </section>
    {:else if activeTab === 'ml-reports' && canModerate}
      <section class="card panel">
        <div class="panel-header">
          <div>
            <h2>ML reports</h2>
            <p class="muted">Reports routed to ML, escalated by ML, or already resolved with ML involvement.</p>
          </div>
          <div class="segmented compact">
            <button type="button" class:active={mlReportsFilter === 'all'} on:click={() => setMlReportsFilter('all')}>
              All
            </button>
            <button
              type="button"
              class:active={mlReportsFilter === 'queued_for_ml'}
              on:click={() => setMlReportsFilter('queued_for_ml')}
            >
              Queued
            </button>
            <button
              type="button"
              class:active={mlReportsFilter === 'under_review'}
              on:click={() => setMlReportsFilter('under_review')}
            >
              Under review
            </button>
            <button
              type="button"
              class:active={mlReportsFilter === 'resolved'}
              on:click={() => setMlReportsFilter('resolved')}
            >
              Resolved
            </button>
          </div>
        </div>

        {#if loadingReports}
          <div class="muted">Loading ML reports...</div>
        {:else if mlReports.length === 0}
          <div class="empty-state">
            <strong>No ML reports</strong>
            <p>Nothing routed to ML matches this filter right now.</p>
          </div>
        {:else}
          <div class="report-list">
            {#each mlReports as report (report.id)}
              <article class="report-card">
                <div class="report-card-header">
                  <div>
                    <h3>{report.reason}</h3>
                    <p class="muted">
                      Status: {report.status} · verdict by {getVerdictActorLabel(report)} · confidence {getConfidenceLabel(report)}
                    </p>
                  </div>
                  <span class={`chip ${report.decision_source === 'ml_auto' ? 'chip-canary' : 'chip-soft'}`}>
                    {report.decision_source === 'ml_auto' ? 'ML verdict' : report.status}
                  </span>
                </div>

                <div class="report-meta-grid">
                  <div>
                    <span class="muted">ML score</span>
                    <strong>{getConfidenceLabel(report)}</strong>
                  </div>
                  <div>
                    <span class="muted">ML verdict</span>
                    <strong>{report.ml_verdict || 'n/a'}</strong>
                  </div>
                  <div>
                    <span class="muted">Model version</span>
                    <strong>{report.ml_model_version || 'n/a'}</strong>
                  </div>
                  <div>
                    <span class="muted">Model stage</span>
                    <strong>
                      {#if report.ml_model_stage}
                        <span class={`chip ${getMlModelStageChipClass(report)}`}>{getMlModelStageLabel(report)}</span>
                      {:else}
                        n/a
                      {/if}
                    </strong>
                  </div>
                  <div>
                    <span class="muted">Scored at</span>
                    <strong>{report.ml_scored_at ? formatDate(report.ml_scored_at) : 'n/a'}</strong>
                  </div>
                  <div>
                    <span class="muted">Decision by</span>
                    <strong>{getVerdictActorLabel(report)}</strong>
                  </div>
                </div>

                <div class="reported-comment">
                  <span class="muted tiny">Reported comment</span>
                  <p>{report.comment_body}</p>
                </div>

                <div class="model-meta">
                  <span>Reporter {report.reporter_name}</span>
                  <span>Author {report.comment_author_name}</span>
                  {#if report.reviewed_by_name}
                    <span>Reviewed by {report.reviewed_by_name}</span>
                  {/if}
                </div>
              </article>
            {/each}
          </div>
        {/if}
      </section>
    {:else if activeTab === 'users' && canAdmin}
      <section class="card panel">
        <div class="panel-header">
          <h2>Users</h2>
          <button type="button" class="ghost-button" on:click={() => loadUsers()}>Refresh</button>
        </div>

        {#if loadingUsers}
          <div class="muted">Loading users...</div>
        {:else if users.length === 0}
          <div class="empty-state">
            <strong>No users found</strong>
            <p>The user list is empty.</p>
          </div>
        {:else}
          <div class="user-list">
            {#each users as user (user.id)}
              <article class="user-card">
                <div class="user-card-main">
                  <div>
                    <h3>{user.display_name || user.username}</h3>
                    <p class="muted">@{user.username} · {user.email || 'no email'}</p>
                  </div>
                  <span class="chip chip-soft">{user.role}</span>
                </div>

                <div class="user-stats">
                  <span>Posts {user.posts_count}</span>
                  <span>Comments {user.comments_count}</span>
                  <span>Reports {user.reports_count}</span>
                </div>

                <div class="user-role-row">
                  <label>
                    <span>Role</span>
                    <select
                      value={userRoleDrafts[user.id] || user.role}
                      disabled={user.id === me.id}
                      on:change={(event) => setUserRoleDraft(user.id, event.currentTarget.value)}
                    >
                      <option value="user">User</option>
                      <option value="moderator">Moderator</option>
                      <option value="admin">Admin</option>
                    </select>
                  </label>

                  <button
                    type="button"
                    class="primary-button"
                    disabled={updatingUserRole === user.id || user.id === me.id}
                    on:click={() => saveUserRole(user)}
                  >
                    {updatingUserRole === user.id ? 'Saving...' : 'Save role'}
                  </button>
                </div>

                {#if user.id === me.id}
                  <div class="muted tiny">You cannot change your own role from this screen.</div>
                {/if}
              </article>
            {/each}
          </div>
        {/if}
      </section>
    {:else if activeTab === 'ml' && canAdmin}
      <section class="card panel">
        <div class="panel-header">
          <div>
            <h2>Model registry</h2>
            <p class="muted">Live deployment status, validation state, and manual promotion controls.</p>
          </div>
          <div class="topbar-actions">
            <button type="button" class="ghost-button" on:click={() => loadModels()}>Refresh</button>
            <button
              type="button"
              class="ghost-button danger"
              on:click={rollbackModel}
              disabled={rollingBackModel}
            >
              {rollingBackModel ? 'Rolling back...' : 'Rollback'}
            </button>
          </div>
        </div>

        <div class="ml-summary-grid">
          <div class="summary-card">
            <span class="muted">Model</span>
            <strong>{modelsOverview?.model_name || 'unknown'}</strong>
          </div>
          <div class="summary-card">
            <span class="muted">Production</span>
            <strong>{modelsOverview?.active_production?.version || 'none'}</strong>
          </div>
          <div class="summary-card">
            <span class="muted">Canary</span>
            <strong>{modelsOverview?.active_canary?.version || 'none'}</strong>
          </div>
          <div class="summary-card">
            <span class="muted">Versions</span>
            <strong>{models.length}</strong>
          </div>
        </div>

        {#if loadingModels}
          <div class="muted">Loading models...</div>
        {:else if models.length === 0}
          <div class="empty-state">
            <strong>No models registered yet</strong>
            <p>Train and register the first model to see the registry here.</p>
          </div>
        {:else}
          <div class="model-list">
            {#each models as model (model.id)}
              <article class="model-card">
                <div class="model-card-header">
                  <div>
                    <h3>{model.version}</h3>
                    <p class="muted">{model.model_name} · feature config v{model.feature_config_version}</p>
                  </div>
                  <span class={`chip ${model.status === 'production' ? 'chip-prod' : model.status === 'canary' ? 'chip-canary' : 'chip-soft'}`}>
                    {model.status}
                  </span>
                </div>

                <div class="model-grid">
                  <div>
                    <span class="muted">Traffic</span>
                    <strong>{model.traffic_percent}%</strong>
                  </div>
                  <div>
                    <span class="muted">Validation acc</span>
                    <strong>{model.validation_accuracy ?? 'n/a'}</strong>
                  </div>
                  <div>
                    <span class="muted">Required acc</span>
                    <strong>{model.required_validation_accuracy}</strong>
                  </div>
                  <div>
                    <span class="muted">Samples</span>
                    <strong>{model.validation_sample_size}</strong>
                  </div>
                </div>

                <div class="model-artifacts">
                  <div>
                    <span class="muted">Artifact</span>
                    <p>{model.artifact_uri}</p>
                  </div>
                  {#if model.metadata_uri}
                    <div>
                      <span class="muted">Metadata</span>
                      <p>{model.metadata_uri}</p>
                    </div>
                  {/if}
                </div>

                {#if model.notes}
                  <p class="muted">{model.notes}</p>
                {/if}

                <div class="model-meta">
                  <span>Created {formatDate(model.created_at)}</span>
                  {#if model.promoted_at}
                    <span>Promoted {formatDate(model.promoted_at)}</span>
                  {/if}
                  {#if model.rolled_back_at}
                    <span>Rolled back {formatDate(model.rolled_back_at)}</span>
                  {/if}
                </div>

                <div class="model-actions">
                  {#if model.status !== 'production' && (model.status === 'canary' || model.status === 'validated' || model.status === 'candidate')}
                    <button
                      type="button"
                      class="primary-button"
                      disabled={promotingModelVersion === model.version}
                      on:click={() => promoteModel(model.version)}
                    >
                      {promotingModelVersion === model.version ? 'Promoting...' : 'Promote'}
                    </button>
                  {/if}
                  {#if model.status === 'production'}
                    <span class="chip chip-prod">Live production</span>
                  {/if}
                  {#if model.status === 'canary'}
                    <span class="chip chip-canary">Canary traffic</span>
                  {/if}
                </div>
              </article>
            {/each}
          </div>
        {/if}
      </section>
    {:else if activeTab === 'profile'}
      <section class="card panel narrow">
        <div class="panel-header">
          <h2>Profile</h2>
        </div>

        <div class="profile-grid">
          <div class="profile-card">
            <span class="muted">Username</span>
            <strong>{currentUserName}</strong>
          </div>
          <div class="profile-card">
            <span class="muted">Role</span>
            <strong>{me.role}</strong>
          </div>
          <div class="profile-card">
            <span class="muted">Posts</span>
            <strong>{me.posts_count}</strong>
          </div>
          <div class="profile-card">
            <span class="muted">Comments</span>
            <strong>{me.comments_count}</strong>
          </div>
          <div class="profile-card">
            <span class="muted">Reports</span>
            <strong>{me.reports_count}</strong>
          </div>
          <div class="profile-card">
            <span class="muted">Hidden comments</span>
            <strong>{me.hidden_comments_count}</strong>
          </div>
        </div>
      </section>
    {/if}
  </main>

  {#if reportTarget}
    <div class="modal-backdrop" role="presentation" on:click={closeReportModal}>
      <div
        class="modal-card"
        role="dialog"
        aria-modal="true"
        aria-labelledby="report-modal-title"
        on:click|stopPropagation
      >
        <div class="panel-header">
          <h3 id="report-modal-title">Report comment</h3>
          <button type="button" class="ghost-button" on:click={closeReportModal}>Close</button>
        </div>

        <p class="muted">
          Reporting {reportTarget.author_name} · {reportTarget.body}
        </p>

        <form class="stack" on:submit|preventDefault={submitReport}>
          <label>
            <span>Reason</span>
            <select bind:value={reportDraft.reason}>
              <option value="harassment">Harassment</option>
              <option value="hate_speech">Hate speech</option>
              <option value="spam">Spam</option>
              <option value="abuse">Abuse</option>
              <option value="other">Other</option>
            </select>
          </label>

          <label>
            <span>Extra note</span>
            <textarea bind:value={reportDraft.reasonText} rows="3" placeholder="Optional context..."></textarea>
          </label>

          <div class="composer-actions">
            <button class="primary-button" type="submit" disabled={creatingReport}>
              {creatingReport ? 'Sending...' : 'Send report'}
            </button>
            <button type="button" class="ghost-button" on:click={closeReportModal}>Cancel</button>
          </div>
        </form>
      </div>
    </div>
  {/if}

  {#if replyTarget}
    <div class="modal-backdrop" role="presentation" on:click={closeReplyModal}>
      <div
        class="modal-card"
        role="dialog"
        aria-modal="true"
        aria-labelledby="reply-modal-title"
        on:click|stopPropagation
      >
        <div class="panel-header">
          <h3 id="reply-modal-title">Reply to {replyTarget.author_name}</h3>
          <button type="button" class="ghost-button" on:click={closeReplyModal}>Close</button>
        </div>

        <p class="muted">{replyTarget.body}</p>

        <form class="stack" on:submit|preventDefault={submitReply}>
          <label>
            <span>Your reply</span>
            <textarea bind:value={replyBody} rows="4" placeholder="Write a reply..."></textarea>
          </label>

          <div class="composer-actions">
            <button class="primary-button" type="submit" disabled={creatingComment}>
              {creatingComment ? 'Sending...' : 'Send reply'}
            </button>
            <button type="button" class="ghost-button" on:click={closeReplyModal}>Cancel</button>
          </div>
        </form>
      </div>
    </div>
  {/if}
{/if}
