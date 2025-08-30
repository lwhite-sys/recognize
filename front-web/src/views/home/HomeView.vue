<template>
  <div class="home">
    <main>
      <!-- Banner区域 -->
      <section class="bg-gradient-soft">
        <div class="container mx-auto px-4 sm:px-6 lg:px-8">
          <div class="flex flex-col md:flex-row items-center">
            <div class="md:w-1/2 mb-10 md:mb-0">
              <h1 class="title text-36 font-bold leading-tight text-gray-800 mb-4 text-shadow">
                在这里，把"成长难题"变成"成长底气"
              </h1>
              <p class="text-[clamp(1rem,2vw,1.25rem)] text-gray-600 mb-8 max-w-lg">
                情绪有处疏、专注不被扰、霸凌能隔绝、关系更融洽——你的「全方位支持站」
              </p>
              <router-link to="/emotional-lightning" class="inline-block bg-primary hover:bg-primary/90 text-white font-medium px-6 py-3 rounded-full shadow-md hover:shadow-lg transition-custom transform hover:-translate-y-1">
                开始探索 <i class="fa fa-arrow-right ml-2"></i>
              </router-link>
            </div>
            <div class="md:w-1/2 flex justify-center">
              <div class="relative w-full max-w-md">
                <!-- 主图形 -->
                <div class="w-42 h-42 md:w-53 md:h-53 bg-white rounded-full shadow-lg flex items-center justify-center relative overflow-hidden">
                  <div class="absolute inset-0 bg-gradient-to-br from-primary/10 to-secondary/10"></div>
                  
                  <!-- 中心人物剪影 -->
                  <div class="w-16 h-24 md:w-21 md:h-32 border-4 border-gray-300 rounded-full relative z-10"></div>
                  
                  <!-- 情绪避雷针图标 -->
                  <div class="absolute top-3 left-3 w-10 h-10 md:w-13 md:h-13 bg-emotion/20 rounded-full flex items-center justify-center animate-pulse">
                    <i class="fa fa-bolt text-emotion text-xl md:text-2xl"></i>
                  </div>
                  
                  <!-- 专注力防护盾图标 -->
                  <div class="absolute top-3 right-3 w-10 h-10 md:w-13 md:h-13 bg-focus/20 rounded-full flex items-center justify-center animate-pulse" style="animation-delay: 0.3s">
                    <i class="fa fa-lightbulb-o text-focus text-xl md:text-2xl"></i>
                  </div>
                  
                  <!-- 霸凌防火墙图标 -->
                  <div class="absolute bottom-3 left-3 w-10 h-10 md:w-13 md:h-13 bg-bullying/20 rounded-full flex items-center justify-center animate-pulse" style="animation-delay: 0.6s">
                    <i class="fa fa-shield text-bullying text-xl md:text-2xl"></i>
                  </div>
                  
                  <!-- 关系保护伞图标 -->
                  <div class="absolute bottom-3 right-3 w-10 h-10 md:w-13 md:h-13 bg-relationship/20 rounded-full flex items-center justify-center animate-pulse" style="animation-delay: 0.9s">
                    <i class="fa fa-users text-relationship text-xl md:text-2xl"></i>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <!-- 功能卡片区域 -->
      <section id="features" class=" bg-neutral">
        <div class="container mx-auto px-4 sm:px-6 lg:px-8">
          <div class="text-center mb-4">
            <h2 class="title1 font-bold text-gray-800 mb-4">全方位成长支持</h2>
            <p class="text-gray-600 max-w-2xl mx-auto text-lg">我们提供四大核心功能，为你的成长之路保驾护航</p>
          </div>
          
          <div class="grid grid-cols-4 gap-4 lg:gap-6">
            <!-- 情绪避雷针 -->
            <div class="custom-card">
              <div class="icon-container bg1"> 
                <i class="fa fa-bolt text-emotion text-2xl"></i>
              </div>
              <h3 class="text-xl font-bold mb-3 text-emotion color1">情绪避雷针</h3>
              <p class="text-gray-600 mb-3">识别烦躁、焦虑的"信号"，给你简单的疏导小技巧，不让坏情绪"炸毛"</p>
              <router-link to="/emotional-lightning" 
              class="color-primary inline-block hover:text-emotion/80 font-medium transition-custom">

                立即进入 <i class="fa fa-arrow-right ml-1"></i>
              </router-link>
            </div>
            
            <!-- 专注力防护盾 -->
            <div class="custom-card">
              <div class="icon-container bg2">
                <i class="fa fa-lightbulb-o text-focus text-2xl"></i>
              </div>
              <h3 class="text-xl font-bold mb-3 text-focus color2">专注力防护盾</h3>

              <p class="text-gray-600 mb-3">挡住"分心小干扰"（比如手机、杂念），教你快速进入"专注模式"</p>
              <router-link to="/concentration-shield" class="color-primary inline-block hover:text-focus/80 font-medium transition-custom">
                立即进入 <i class="fa fa-arrow-right ml-1"></i>
              </router-link>
            </div>
            
            <!-- 霸凌防火墙 -->
            <div class="custom-card">
              <div class="icon-container bg3">
                <i class="fa fa-shield text-bullying text-2xl"></i>
              </div>
              <h3 class="text-xl font-bold mb-3 text-bullying color3">霸凌防火墙</h3>
              <p class="text-gray-600 mb-3">教你识别霸凌行为，提供应对方法+求助路径，让"恶意"进不来</p>
              <router-link to="/bullying-firewall" 
              class="color-primary inline-block hover:text-bullying/80 font-medium transition-custom">
                立即进入 <i class="fa fa-arrow-right ml-1"></i>
              </router-link>
            </div>
            
            <!-- 关系保护伞 -->
            <div class="custom-card">
              <div class="icon-container bg4">
                <i class="fa fa-users text-relationship text-2xl"></i>
              </div>
              <h3 class="text-xl font-bold mb-3 color4">关系保护伞</h3>
              <p class="text-gray-600 mb-3">解决亲子、同学间的小矛盾，给你"好好沟通"的小妙招，让关系更舒服</p>
              <router-link to="/relationship-umbrella" class="color-primary inline-block hover:text-relationship/80 font-medium transition-custom">
                立即进入 <i class="fa fa-arrow-right ml-1"></i>
              </router-link>
            </div>
          </div>
        </div>
      </section>
      
      <!-- 信任区 -->
      <section class="pt-4 bg-white mt-10">
        <div class="container mx-auto px-4 sm:px-6 lg:px-8">
          <div class="text-center">
            <h2 class="title1 font-bold text-gray-800 mb-4">谁在使用这里？</h2>
          </div>
          
          <div class="grid grid-cols-3 gap-8">
            <div class="text-center p-4 rounded-xl bg-neutral">
              <div class="w-16 h-16 bg-primary/20 rounded-full flex items-center justify-center mx-auto mb-4">
                <i class="fa fa-user text-primary text-2xl"></i>
              </div>
              <h3 class="font-medium text-lg mb-2">需要梳理情绪的你</h3>
              <p class="text-gray-600">无论你是学生还是青少年，这里都能帮助你更好地理解和管理情绪</p>
            </div>
            
            <div class="text-center p-4 rounded-xl bg-neutral">
              <div class="w-16 h-16 bg-secondary/20 rounded-full flex items-center justify-center mx-auto mb-4 bg5">
                <i class="fa fa-home text-secondary text-2xl "></i>
              </div>
              <h3 class="font-medium text-lg mb-2">想帮孩子的家长</h3>
              <p class="text-gray-600">提供专业的指导，帮助家长更好地支持孩子面对成长中的各种挑战</p>
            </div>
            
            <div class="text-center p-4 rounded-xl bg-neutral">
              <div class="w-16 h-16 bg-primary/20 rounded-full flex items-center justify-center mx-auto mb-4">
                <i class="fa fa-briefcase text-primary text-2xl"></i>
              </div>
              <h3 class="font-medium text-lg mb-2">关注学生的老师</h3>
              <p class="text-gray-600">为教育工作者提供工具和资源，更好地关注学生的心理健康和成长</p>
            </div>
          </div>
        </div>
      </section>
    </main>

  
  </div>
</template>

<script setup>
</script>

<style scoped>
.custom-card {
  background-color: white;
  border-radius: 0.75rem;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
  padding: 1rem;
  transition: all 0.3s ease;
  transform: translateY(0);
}

.custom-card:hover {
  box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
  transform: translateY(-0.5rem);
}
.icon-container{
  width: 4rem;
  height: 4rem;
  border-radius: 9999px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 0.5rem;
}
.bg1 {
  background-color: rgb(255 183 77 / 0.2);
}

.bg2 {
  background-color:rgb(102 187 106 / 0.2);
}
.bg3 {
  background-color: rgb(239 83 80 / 0.2);
}
.bg4 {
  background-color:rgb(255 238 88 / 0.2);
}

.color1{
color: #FFB74D;
}
.color2{
  color: #66BB6A;
}
.color3{
  color: #EF5350;
}
.color4{
  color: #FFEE58;
}
.bg5{
  background-color: rgb(129 199 132 / 0.2);
}
.title{
  font-size: 30px;

}
.title1{
  font-size: 24px;
}
.text-emotion {
  color: #FFB74D;
  font-size: 24px;

}
.text-focus {
    font-size: 24px;
   color: #66BB6A;
}
.text-bullying {
    font-size: 24px;
 color: #EF5350;
}
.text-relationship {
    font-size: 24px;
  color: #FFEE58;
}

.mt-10{
  margin-top: 10px;

}
.p-4 {
  padding: 4px;
}
.color-primary{
  color: #3b82f6;
}
.home {
  font-family: 'Inter', sans-serif;
}

.sticky {
  position: sticky;
}

.top-0 {
  top: 0;
}

.z-50 {
  z-index: 50;
}

.bg-white\/95 {
  background-color: rgba(255, 255, 255, 0.95);
}

.shadow-sm {
  box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
}

.backdrop-blur-sm {
  backdrop-filter: blur(4px);
}

.transition-custom {
  transition: all 0.3s ease;
}

.container {
  width: 100%;
  margin-left: auto;
  margin-right: auto;
}

.mx-auto {
  margin-left: auto;
  margin-right: auto;
}

.px-4 {
  padding-left: 1rem;
  padding-right: 1rem;
}

.sm\:px-6 {
  @media (min-width: 640px) {
    padding-left: 1.5rem;
    padding-right: 1.5rem;
  }
}

.lg\:px-8 {
  @media (min-width: 1024px) {
    padding-left: 2rem;
    padding-right: 2rem;
  }
}

.flex {
  display: flex;
}

.justify-between {
  justify-content: space-between;
}

.items-center {
  align-items: center;
}

.h-16 {
  height: 4rem;
}

.md\:h-20 {
  @media (min-width: 768px) {
    height: 5rem;
  }
}

.w-10 {
  width: 2.5rem;
}

.h-10 {
  height: 2.5rem;
}

.md\:w-12 {
  @media (min-width: 768px) {
    width: 3rem;
  }
}

.md\:h-12 {
  @media (min-width: 768px) {
    height: 3rem;
  }
}

.rounded-full {
  border-radius: 9999px;
}

.bg-gradient-to-r {
  background-image: linear-gradient(to right, var(--tw-gradient-stops));
}

.from-primary {
  --tw-gradient-from: #3b82f6;
  --tw-gradient-stops: var(--tw-gradient-from), var(--tw-gradient-to, rgba(59, 130, 246, 0));
}

.to-secondary {
  --tw-gradient-to: #10b981;
}

.mr-3 {
  margin-right: 0.75rem;
}

.text-white {
  color: white;
}

.text-xl {
  font-size: 1.25rem;
  line-height: 1.75rem;
}

.md\:text-2xl {
  @media (min-width: 768px) {
    font-size: 1.5rem;
    line-height: 2rem;
  }
}

.font-bold {
  font-weight: 700;
}

.bg-clip-text {
  -webkit-background-clip: text;
  background-clip: text;
}

.text-transparent {
  color: transparent;
}

.hidden {
  display: none;
}

.md\:flex {
  @media (min-width: 768px) {
    display: flex;
  }
}

.space-x-8 > * + * {
  margin-left: 2rem;
}

.text-gray-600 {
  color: #4b5563;
}

.hover\:text-primary:hover {
  color: #3b82f6;
}

.py-5 {
  padding-top: 1.25rem;
  padding-bottom: 1.25rem;
}

.border-b-2 {
  border-bottom-width: 2px;
}

.border-primary {
  border-color: #3b82f6;
}

.px-1 {
  padding-left: 0.25rem;
  padding-right: 0.25rem;
}

.bg-emotion {
  --emotion: #f59e0b;
  color: var(--emotion);
}

.bg-emotion\/20 {
  background-color: rgba(245, 158, 11, 0.2);
}

.bg-focus {
  --focus: #10b981;
  color: var(--focus);
}

.bg-focus\/20 {
  background-color: rgba(16, 185, 129, 0.2);
}

.bg-bullying {
  --bullying: #ef4444;
  color: var(--bullying);
}

.bg-bullying\/20 {
  background-color: rgba(239, 68, 68, 0.2);
}

.bg-relationship {
  --relationship: #8b5cf6;
  color: var(--relationship);
}

.bg-relationship\/20 {
  background-color: rgba(139, 92, 246, 0.2);
}

.bg-neutral {
  background-color: #f3f4f6;
}

.shadow-card {
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}

.hover\:shadow-card-hover:hover {
  box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
}

.transform {
  transform: translate(var(--tw-translate-x), var(--tw-translate-y)) rotate(var(--tw-rotate)) skewX(var(--tw-skew-x)) skewY(var(--tw-skew-y)) scaleX(var(--tw-scale-x)) scaleY(var(--tw-scale-y));
}

.hover\:-translate-y-1:hover {
  transform: translateY(-0.25rem);
}

.hover\:-translate-y-2:hover {
  transform: translateY(-0.5rem);
}

.animate-pulse {
  animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
}

@keyframes pulse {
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.5;
  }
}

.text-shadow {
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
}

.max-w-lg {
  max-width: 32rem;
}

.max-w-md {
  max-width: 28rem;
}

.max-w-2xl {
  max-width: 42rem;
}

.grid {
  display: grid;
}

.grid-cols-1 {
  grid-template-columns: repeat(1, minmax(0, 1fr));
}

.md\:grid-cols-2 {
  @media (min-width: 768px) {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

.gap-8 {
  gap: 2rem;
}

.lg\:gap-10 {
  @media (min-width: 1024px) {
    gap: 2.5rem;
  }
}

.p-4 {
  padding: 1.5rem;
}

.mb-6 {
  margin-bottom: 1.5rem;
}

.mb-4 {
  margin-bottom: 1rem;
}

.mb-3 {
  margin-bottom: 0.75rem;
}

.mb-8 {
  margin-bottom: 2rem;
}

.mb-12 {
  margin-bottom: 3rem;
}

.text-lg {
  font-size: 1.125rem;
  line-height: 1.75rem;
}

.text-center {
  text-align: center;
}

.mx-auto {
  margin-left: auto;
  margin-right: auto;
}

.bg-white {
  background-color: white;
}

.bg-gradient-soft {
  background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%);
}

.py-16 {
  padding-top: 4rem;
  padding-bottom: 4rem;
}

.md\:py-24 {
  @media (min-width: 768px) {
    padding-top: 6rem;
    padding-bottom: 6rem;
  }
}

.leading-tight {
  line-height: 1.25;
}

.text-gray-800 {
  color: #1f2937;
}

.inline-block {
  display: inline-block;
}

.bg-primary {
  --primary: #3b82f6;
  background-color: var(--primary);
}

.hover\:bg-primary\/90:hover {
  background-color: rgba(59, 130, 246, 0.9);
}

.rounded-full {
  border-radius: 9999px;
}

.shadow-md {
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}

.hover\:shadow-lg:hover {
  box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
}

.ml-2 {
  margin-left: 0.5rem;
}

.ml-1 {
  margin-left: 0.25rem;
}

.relative {
  position: relative;
}

.w-full {
  width: 100%;
}

.overflow-hidden {
  overflow: hidden;
}

.absolute {
  position: absolute;
}

.inset-0 {
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;
}

.from-primary\/10 {
  --tw-gradient-from: rgba(59, 130, 246, 0.1);
  --tw-gradient-stops: var(--tw-gradient-from), var(--tw-gradient-to, rgba(59, 130, 246, 0));
}

.to-secondary\/10 {
  --tw-gradient-to: rgba(16, 185, 129, 0.1);
}

.z-10 {
  z-index: 10;
}

.border-4 {
  border-width: 4px;
}

.border-gray-300 {
  border-color: #d1d5db;
}

.top-5 {
  top: 1.25rem;
}

.left-5 {
  left: 1.25rem;
}

.right-5 {
  right: 1.25rem;
}

.bottom-5 {
  bottom: 1.25rem;
}

.w-16 {
  width: 4rem;
}

.h-16 {
  height: 4rem;
}

.md\:w-20 {
  @media (min-width: 768px) {
    width: 5rem;
  }
}

.md\:h-20 {
  @media (min-width: 768px) {
    height: 5rem;
  }
}

.flex {
  display: flex;
}

.items-center {
  align-items: center;
}

.justify-center {
  justify-content: center;
}

.bg-gray-50 {
  background-color: #f9fafb;
}

.border-t {
  border-top-width: 1px;
}

.border-gray-200 {
  border-color: #e5e7eb;
}

.py-10 {
  padding-top: 2.5rem;
  padding-bottom: 2.5rem;
}

.text-sm {
  font-size: 0.875rem;
  line-height: 1.25rem;
}

.text-gray-500 {
  color: #6b7280;
}

:root {
  --primary: #3b82f6;
  --secondary: #10b981;
  --emotion: #f59e0b;
  --focus: #10b981;
  --bullying: #ef4444;
  --relationship: #8b5cf6;
}
</style>