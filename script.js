
const data = [
  { 
    interest :[
      "medicine",
      "doctor",
      "medical science",
      "clinical medicine",
      "patient care",
      "hospital work"
    ],
    majors: ['Neuroscience', 'Biochemistry', 'Anatomy', 'Physiology','Pharmacology', 'Microbiology', 'Immunology', 'Pathology', 'Biostatistics', 'Health Informatics', 'Medical Ethics', 'Public Health', 'Epidemiology', 'Biomedical Science','Medical Biotechnology'],
    colleges:['University of California, San Francisco','Johns Hopkins University', 'University of Oxford', 'Peking University Health Science Center','University of Sydney',"Monash University","Heidelberg University","University of Amsterdam","University of Toronto","McGill University",'Imperial College London'],
  },

  {
    interest: [
      "art",
      "visual arts",
      "painting and sketching",
      "graphic design",
      "creative illustration",
      "fine arts practice"
    ],
    majors: ['Fine Arts', 'Graphic Design', 'Painting', 'Sculpture', 'Drawing', 'Printmaking', 'Photography', 'Digital Art', 'Ceramics', 'Textile Art', 'Fashion Design', 'Interior Design', 'Architecture', 'Landscape Architecture', 'Urban Planning', 'Illustration', 'Animation', 'Film Studies', 'Theatre Arts', 'Visual Communication'],
    colleges: ['Rhode Island School of Design', 'Royal College of Art', 'Accademia di Belle Arti di Firenze','Ontario College of Art and Design', 'Savannah College of Art and Design', 'Central Acadamy of Fine Arts','RMIT University','School of the Art Institute of Chicago','California Institute of the Arts'],
  },

  {
    interest: [
      "space",
      "astronomy",
      "space exploration",
      "rocket science",
      "aerospace engineering",
      "planetary science research"
    ],
    majors: ['Astronomy', 'Astrophysics', 'Planetary Science', 'Space Science', 'Aerospace Engineering', 'Aeronautical Engineering', 'Astrobiology', 'Cosmology', 'Space Physics', 'Space Weather', 'Space Systems Engineering', 'Spacecraft Engineering', 'Spacecraft Propulsion','Spacecraft Guidance and Control','Spacecraft Telecommunications','Spacecraft Avionics','Spacecraft Command and Data Handling'],
    colleges: ['Massachusetts Institute of Technology', 'California Institute of Technology','University of colorado boulder','Embry-Riddle Aeronautical University','Georgia Institute of Technology','University of Cambrigde','University of Leicester','University of Toronto','Technical University of Munich','ISAE-SUPAERO','University of Sydney'],
  },

  {
    interest: [
      "coding",
      "software engineering",
      "computer programming",
      "web development",
      "app development",
      "computer science engineering"
    ],
    majors: ['Computer Science', 'Software Engineering', 'Information Technology', 'Data Science', 'Artificial Intelligence', 'Machine Learning', 'Cybersecurity', 'Computer Engineering', 'Robotics', 'Game Development', 'Web Development', 'Mobile App Development', 'Database Management', 'Networking', 'Operating Systems', 'Computer Graphics', 'Human-Computer Interaction', 'Computer Vision', 'Natural Language Processing', 'Computer Architecture'],
    colleges: ['Massachusetts Institute of Technology', 'Georgia Institute of Technology','Cornell University','University of California, San Diego','University of Texas at Austin','University of Illinois at Urbana-Champaign','University of Washington','Imperial College London','Mc Gill University','Australia National University','Shanghai Jiao Tong University','University of Oxford','University of Sydney'],
  },

  {
    interest: [
      "law",
      "legal studies",
      "criminal law",
      "corporate law",
      "constitutional law",
      "legal research and justice system"
    ],
    majors: ['Criminal Law', 'Civil Law', 'International Law', 'Tax Law', 'Corporate Law', 'Intellectual Property Law', 'Labor Law', 'Health Law', 'Human Rights Law', 'Environmental Law', 'Entertainment Law', 'Banking Law', 'Insurance Law', 'Real Estate Law','Constitutional Law','Administrative Law'],
    colleges: ['University of Oxford','University of Cambridge','University of California, Berkeley','University of Chicago','University of Michigan','University of Toronto','University of Sydney','University of Melbourne','University of Amsterdam','University of Tokyo','New York University','Colombia University'],
  },

  {
    interest: [
      "music",
      "music performance",
      "singing and vocal training",
      "instrumental music",
      "music production",
      "music composition and theory"
    ],
    majors: ['Music Theory', 'Music History', 'Music Composition', 'Music Performance', 'Music Education', 'Music Technology', 'Music Production', 'Music Business', 'Music Industry', 'Music Journalism', 'Music Therapy', 'Music Psychology', 'Music Anthropology', 'Music Sociology', 'Music Pedagogy', 'Music Ethnomusicology', 'Music Acoustics', 'Music Cognition', 'Music Aesthetics', 'Music Philosophy'],
    colleges: ['Berklee College of Music','Juilliard School','Royal College of Music (United Kingdom)','Curtis Institute of Music','New England Conservatory','University of Southern California','Conservatoire de Paris','Conservatorio di Musica Giuseppe Verdi (Milan)'],
  }
];

const vocab = [
  'doctor','medicine','hospital','medical','biology','patient','health',
  'code','app','software','program','coding','developer','computer',
  'law','lawyer','legal','court','judge','justice','case',
  'draw','art','painting','sketch','design','creative',
  'music','sing','song','guitar','piano','band',
  'space','rocket','astronomy','planet','nasa','universe'
];

// ------------------------------
// TF-IDF HELPERS
// ------------------------------
function tokenize(text) {
  return text.toLowerCase().split(/[^a-z]+/).filter(Boolean);
}

function tf(tokens, term) {
  let count = tokens.filter(t => t === term).length;
  return count / (tokens.length || 1);
}

function idf(corpus, term) {
  let docsWith = 0;

  corpus.forEach(doc => {
    if (doc.tokens.includes(term)) docsWith++;
  });

  return Math.log((corpus.length + 1) / (docsWith + 1)) + 1;
}

// ------------------------------
// BUILD CORPUS
// ------------------------------
function buildCorpus() {
  let corpus = [];

  data.forEach((item, idx) => {
    item.interest.forEach(text => {
      corpus.push({
        text,
        label: idx,
        tokens: tokenize(text)
      });
    });
  });

  return corpus;
}

const corpus = buildCorpus();


function vectorize(text) {
  let tokens = tokenize(text);

  return vocab.map(term => {
    return tf(tokens, term) * idf(corpus, term);
  });
}

function cosine(a, b) {
  let dot = 0, magA = 0, magB = 0;

  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] ** 2;
    magB += b[i] ** 2;
  }

  return dot / (Math.sqrt(magA) * Math.sqrt(magB) + 1e-10);
}


function predict(text) {
  let inputVec = vectorize(text);

  let scores = data.map((_, i) => {

    let maxSim = 0;

    corpus.forEach(doc => {
      if (doc.label === i) {
        let docVec = vectorize(doc.text);
        let sim = cosine(inputVec, docVec);

        if (sim > maxSim) maxSim = sim;
      }
    });

    return maxSim;
  });

  let best = scores.indexOf(Math.max(...scores));
  let confidence = scores[best];
  confidence = Math.min(confidence * 1.5, 1);

  if (confidence < 0.08) {
    return { label: "unknown", confidence: 0 };
  }

  return {
    label: data[best].interest[0],
    confidence
  };
}

function findColleges() {
  const input = document.getElementById("interest").value;
  const resultDiv = document.getElementById("results");

  const result = predict(input);

  const match = data.find(d => d.interest.includes(result.label));

  if (match) {
    resultDiv.innerHTML = `
      <h2>Predicted Field: ${result.label}</h2>
      <p>Confidence: ${(result.confidence * 100).toFixed(2)}%</p>

      <h3>Majors:</h3>
      <ul>${match.majors.map(m => `<li>${m}</li>`).join("")}</ul>

      <h3>Colleges:</h3>
      <ul>${match.colleges.map(c => `<li>${c}</li>`).join("")}</ul>
    `;
  } else {
    resultDiv.innerHTML = `
      <h2>No strong match found</h2>
      <p>Try rephrasing your interest.</p>
    `;
  }
}

window.findColleges = findColleges;
