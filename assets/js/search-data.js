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
        },{id: "nav-cv",
          title: "CV",
          description: "This is a description of the page. You can modify it in &#39;_pages/cv.md&#39;. You can also change or remove the top pdf download button.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
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
      },{
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
