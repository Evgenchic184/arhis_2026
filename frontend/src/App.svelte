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

  let posts = [];
  let selectedPost = null;
  let comments = [];
  let commentTree = [];
  let reports = [];
  let users = [];

  let replyTarget = null;
  let reportTarget = null;
  let userRoleDrafts = {};

  let authBusy = false;
  let loadingPosts = false;
  let loadingComments = false;
  let loadingReports = false;
  let loadingUsers = false;
  let creatingPost = false;
  let creatingComment = false;
  let creatingReport = false;
  let updatingUserRole = '';
  let decidingReportId = '';

  let authError = '';
  let appError = '';
  let notice = '';

  $: canModerate = Boolean(me && (me.role === 'moderator' || me.role === 'admin'));
  $: canAdmin = Boolean(me && me.role === 'admin');
  $: currentUserName = me ? (me.display_name || me.username) : '';

  onMount(async () => {
    token = localStorage.getItem(storageKey) || '';
    if (token) {
      await restoreSession();
    }
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
    users = [];
    replyTarget = null;
    reportTarget = null;
    userRoleDrafts = {};
    commentBody = '';
    replyBody = '';
    localStorage.removeItem(storageKey);
  }

  async function restoreSession() {
    try {
      me = await api.me(token);
      await loadPosts({ silent: true });
      if (me && (me.role === 'moderator' || me.role === 'admin')) {
        await loadReports({ silent: true });
      }
      if (me && me.role === 'admin') {
        await loadUsers({ silent: true });
      }
    } catch (error) {
      clearSession();
      setError(error.message || 'Session expired. Please sign in again.');
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
      }
      if (me && me.role === 'admin') {
        await loadUsers({ silent: true });
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

  async function selectPost(post) {
    selectedPost = post;
    replyTarget = null;
    reportTarget = null;
    commentTree = [];
    commentBody = '';
    replyBody = '';

    try {
      loadingComments = true;
      const freshPost = await api.getPost(post.id);
      selectedPost = freshPost;

      const data = await api.listComments(post.id);
      comments = Array.isArray(data) ? data : [];
      commentTree = buildCommentTree(comments);
    } catch (error) {
      setError(error.message || 'Failed to load comments.');
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

  async function refreshSelectedPost() {
    if (!selectedPost) {
      return;
    }

    await loadPosts({ silent: true });
    await selectPost(selectedPost);
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
      const data = await api.listReports(token, status);
      reports = Array.isArray(data) ? data : [];
    } catch (error) {
      setError(error.message || 'Failed to load reports.');
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

  async function setReportsFilter(nextFilter) {
    reportsFilter = nextFilter;
    await loadReports();
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
    if (tab === 'users' && canAdmin) {
      loadUsers();
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
      {/if}
      {#if canAdmin}
        <button type="button" class:active={activeTab === 'users'} on:click={() => switchTab('users')}>
          Users
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
            <p>The queue is empty right now.</p>
          </div>
        {:else}
          <div class="report-list">
            {#each reports as report (report.id)}
              <article class="report-card">
                <div class="report-card-header">
                  <div>
                    <h3>{report.reason}</h3>
                    <p class="muted">
                      Status: {report.status} · comment author {report.comment_author_name} · reporter {report.reporter_name}
                    </p>
                  </div>
                  {#if report.moderation_verdict}
                    <span class="chip chip-soft">{report.moderation_verdict}</span>
                  {/if}
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
