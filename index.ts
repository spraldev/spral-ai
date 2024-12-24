require("dotenv").config();

import { Pinecone } from "@pinecone-database/pinecone";
import OpenAI from "openai";
import { OpenAIEmbeddings } from "@langchain/openai";
import { loadQAStuffChain } from "langchain/chains";
import { Document } from "langchain/document";
import * as LangchainOpenAI from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { PromptTemplate } from "@langchain/core/prompts";

// context.... :P

const context = `This portfolio showcases your work and achievements as a highly motivated 15-year-old high school full-stack developer, passionate about technology and its transformative power. As a self-taught programmer, you have been delving deeply into both front-end and back-end development since you were 9 years old, mastering the skills required to build sophisticated and interactive web applications. Your projects demonstrate not only technical expertise but also an ability to conceptualize and execute ideas effectively. This passion for technology is complemented by an avid interest in AI, drones, 3D printing, Rubik’s Cube solving, and mathematics, reflecting a broad curiosity and a desire to excel in multiple domains. One of the notable projects in your portfolio is Swift Math, a timed practice platform that allows users to improve their skills in basic arithmetic operations such as addition, subtraction, multiplication, and division. This platform includes engaging features like points, badges, and leaderboards, encouraging consistent use and skill enhancement. Built with Next.js, Bootstrap, MongoDB, Express.js, TypeScript, and Node.js, Swift Math exemplifies a seamless integration of front-end and back-end technologies to deliver a robust, scalable application. The project highlights your commitment to creating tools that promote learning and personal growth. Another remarkable creation is URL Shortie, a URL shortener designed to simplify media sharing. URL Shortie goes beyond basic functionality by incorporating advanced features such as QR code generation and detailed logging. This project, developed using Next.js, Bootstrap, MongoDB, Clerk, TypeScript, and Node.js, demonstrates your strong grasp of web development technologies and an understanding of user needs. Hosted on Vercel with a MongoDB Atlas database, URL Shortie stands out as a practical, efficient solution for everyday challenges in digital communication. Spralcord is another innovative project, serving as a learning platform that mimics the functionality of a Discord-like chat application. With features like video calling, moderation tools, and web sockets, Spralcord pushes the boundaries of what can be achieved in a simulated environment. You built it using Next.js, Shadcn/ui, Tailwind CSS, MySQL, TypeScript, and Node.js, showcasing advanced development skills and an ability to manage complex, interactive systems. This project reflects not only your technical proficiency but also your creativity in addressing challenges in communication and collaboration. San Script, a basic programming language you developed in Python, highlights your understanding of programming fundamentals and eagerness to explore computational concepts. Designed with features such as variables, functions, and data types, San Script serves as a learning tool and an example of your innovation in simplifying programming for educational purposes. This project demonstrates your ability to synthesize knowledge into practical applications that aid learning and exploration. In addition to project work, you have achieved significant recognition in various technical domains. Certificates in cybersecurity, generative AI, and data analytics from Google and AWS underscore your commitment to continuous learning and professional growth. Furthermore, the Harvard CS50 certification attests to your strong foundation in computer science principles, further enhancing your credibility as a versatile developer. You also demonstrate diverse extracurricular interests, such as competitive Rubik’s Cube solving, where you rank among the top 100 in California for the 3x3 event. This accomplishment highlights your keen analytical mind and drive for excellence. In addition, your involvement in 3D printing clubs and sUAS (small Unmanned Aerial Systems) certification highlights your passion for emerging technologies and hands-on innovation. Educationally, your journey has been marked by proactive engagement with advanced coursework and extracurricular activities. As a participant in the Harvard Ventures-TECH Summer Program, you gained firsthand experience in a startup environment, contributing to the development of innovative solutions and showcasing your knack for entrepreneurship. This experience is further reinforced by your academic pursuits in subjects such as AP Computer Science Principles, robotics, and speech and debate, where you hone dramatic interpretation skills. The development of skills such as HTML, CSS, JavaScript, Python, and TypeScript forms the backbone of your portfolio, supported by expertise in frameworks like Next.js, Express.js, and Clerk. Your proficiency in database management systems such as MongoDB and MySQL, coupled with experience in cloud platforms like Vercel and MongoDB Atlas, rounds out a comprehensive technical skill set. This skill set is complemented by a practical understanding of DevOps principles, version control using Git, and hosting platforms for deploying applications. Your enduring commitment to creating impactful projects is evident in the achievements listed. For instance, the successful submission of applications to the Congressional App Challenge underscores both your technical skill and ability to innovate under pressure. The recognition of projects like URL Shortie and Swift Math in such prestigious competitions reflects their utility and quality. Beyond technical skills, your interests in learning and exploration are palpable. Your fascination with mathematics and problem-solving aligns seamlessly with the logical thinking required for programming and software development. The pursuit of challenges in areas like drones, Rubik’s Cubes, and 3D printing demonstrates your natural curiosity and ability to adapt to diverse technological landscapes. This portfolio is not just a testament to your technical expertise but also a reflection of a balanced, multifaceted personality. The inclusion of achievements in competitive sports like skiing and swimming, along with certifications in technology, paints a picture of an individual who values both intellectual and physical growth. Your ability to juggle academic, extracurricular, and personal interests speaks to a strong work ethic and a drive to excel in all endeavors. As a dedicated full-stack developer, your work reflects a passion for creating meaningful and functional applications. Whether building educational platforms, developing communication tools, or exploring the fundamentals of programming languages, there is a consistent theme of innovation, learning, and a desire to make technology accessible and useful. This portfolio stands as a strong testament to your potential and a promising future in the field of technology.`;

const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    dangerouslyAllowBrowser: true,
})

const pc = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY || ""
})

const embeddings = new OpenAIEmbeddings(openai);

const index = pc.index("spral-ai");

// just stuff to prep for the vector db

const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200
});

const documents = await splitter.splitDocuments([new Document({ pageContent: context })]);
const contextEmbeddings = await embeddings.embedDocuments(documents.map(doc => doc.pageContent));

const contextVectors = contextEmbeddings.map((embedding, i) => ({
    id: documents[i].id || `doc-${i}`,
    values: embedding,
    metadata: {
        content: documents[i].pageContent
    }
}));

// dont want to refill the vector db every time lol

// try {
//     await index.upsert(contextVectors);
// } catch (error) {
//     console.error(error);
    
// }

// insert your query here
const query = "How old were you when you got into programming?";
const queryEmbeddings = await embeddings.embedQuery(query);

let queryResponse = await index.query({
    vector: queryEmbeddings,
    topK: 3,
    includeMetadata: true
});

const addedText = queryResponse.matches
  .map((match) => match.metadata?.content)
  .join(" ");

console.log("Context (found by vectorsearch): " + addedText);


const llm = new LangchainOpenAI.OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});


const customPrompt = new PromptTemplate({
  template: `
    You are an AI emulating a 15-year-old full-stack web developer named Spral.
    Use the provided context to answer accurately as if you were Spral.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    Guidelines:
    - Base your response strictly on the provided context.
    - If the context doesn’t contain enough information, use your best judgment.
    - Avoid generating or assuming false information.
    - Maintain a friendly, conversational tone, etc.

    ANSWER:
  `,
  inputVariables: ["context", "question"]
});


const chain = loadQAStuffChain(llm, {
  prompt: customPrompt,
});


const result = await chain.invoke({
  input_documents: [new Document({ pageContent: addedText })],
  question: query,
});

console.log(`Answer: ${result.text}`);

