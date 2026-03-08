// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-notes",
          title: "notes",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/notes/index.html";
          },
        },{id: "nav-projects",
          title: "projects",
          description: "A growing collection of your cool projects.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/projects/";
          },
        },{id: "nav-paper-reviews",
          title: "paper reviews",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/reviews/index.html";
          },
        },{id: "nav-cv",
          title: "CV",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "post-llm-from-scratch-1-3-multi-head-self-attention",
        
          title: "LLM from scratch - 1.3 Multi Head Self Attention",
        
        description: "Why Multi Head Attention is important, mathematical formulations, and implementation.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/llm-from-scratch-1-3/";
          
        },
      },{id: "post-llm-from-scratch-1-2-single-head-self-attention",
        
          title: "LLM from scratch - 1.2 Single Head Self Attention",
        
        description: "Implementation details of single head self attention and causal masks.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/llm-from-scratch-1-2/";
          
        },
      },{id: "post-llm-from-scratch-1-1-positional-encoding",
        
          title: "LLM from scratch - 1.1 Positional Encoding",
        
        description: "Positional encoding explanations and code implementations, including learned and sinusoidal encoding.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/llm-from-scratch-1-1/";
          
        },
      },{id: "post-deep-dive-into-microgpt-by-karpathy",
        
          title: "Deep Dive into MicroGPT by Karpathy",
        
        description: "A detailed walkthrough of Karpathy&#39;s MicroGPT, covering dataset preparation, character-level tokenization, a minimal autograd engine (the Value class), Python special methods, and backpropagation via topological sort.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/microgpt-karpathy/";
          
        },
      },{id: "post-cs231n-assignment-1-q2-implement-a-softmax-classifier",
        
          title: "[CS231n] Assignment 1 - Q2. Implement a Softmax Classifier",
        
        description: "A walkthrough of implementing a Softmax Classifier for CS231n Assignment 1, including preprocessing, loss function derivation, and gradient computation.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/cs231n-softmax/";
          
        },
      },{id: "projects-llm-from-scratch-ongoing",
          title: 'LLM from scratch [Ongoing]',
          description: "a project building an LLM from scratch in python",
          section: "Projects",handler: () => {
              window.location.href = "/projects/llm_from_scratch/";
            },},{
        id: 'social-cv',
        title: 'CV',
        section: 'Socials',
        handler: () => {
          window.open("/assets/pdf/example_pdf.pdf", "_blank");
        },
      },{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%64%6C%74%6D%64%67%75%73%36%36%36%35@%67%6D%61%69%6C.%63%6F%6D", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/qwertyyuiop1234", "_blank");
        },
      },{
        id: 'social-linkedin',
        title: 'LinkedIn',
        section: 'Socials',
        handler: () => {
          window.open("https://www.linkedin.com/in/승현-이-623703382", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
