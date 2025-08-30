<template>
  <NSpace vertical :size="16">
    <!-- <HeaderBanner /> -->
    <!-- <CardData /> -->
    <!-- <NGrid :x-gap="gap" :y-gap="16" responsive="screen" item-responsive>
      <NGi span="24 s:24 m:14">
        <NCard :bordered="false" class="card-wrapper"> -->

    <HomeView />
  
        <!-- </NCard>
      </NGi> -->
      <!-- <NGi span="24 s:24 m:10">
        <NCard :bordered="false" class="card-wrapper">
          <PieChart />
        </NCard>
      </NGi> -->
    <!-- </NGrid> -->
    <!-- <NGrid :x-gap="gap" :y-gap="16" responsive="screen" item-responsive>
      <NGi span="24 s:24 m:14">
        <ProjectNews />
      </NGi>
      <NGi span="24 s:24 m:10">
        <CreativityBanner />
      </NGi>
    </NGrid> -->
  </NSpace>
</template>
<script setup lang="ts">
import { ref, onMounted, watch } from 'vue';
import { useRouter } from 'vue-router';
import HomeView from './HomeView.vue'
import BannerSection from './components/BannerSection.vue'
import FeaturesSection from './components/FeaturesSection.vue'
import TrustSection from './components/TrustSection.vue'
import Footer from './components/Footer.vue'
import { computed } from 'vue';
import { useAppStore } from '@/store/modules/app';
import HeaderBanner from './modules/header-banner.vue';
import CardData from './modules/card-data.vue';
import LineChart from './modules/line-chart.vue';
import PieChart from './modules/pie-chart.vue';
import ProjectNews from './modules/project-news.vue';
import CreativityBanner from './modules/creativity-banner.vue';

const appStore = useAppStore();

const gap = computed(() => (appStore.isMobile ? 0 : 16));
// 状态管理
const isScrolled = ref(false);
const mobileMenuOpen = ref(false);
const router = useRouter();

// 功能数据
const features = [
  {
    icon: 'fa-bolt',
    color: 'emotion',
    title: '情绪避雷针',
    description: '识别烦躁、焦虑的"信号"，给你简单的疏导小技巧，不让坏情绪"炸毛"'
  },
  {
    icon: 'fa-lightbulb-o',
    color: 'focus',
    title: '专注力防护盾',
    description: '挡住"分心小干扰"（比如手机、杂念），教你快速进入"专注模式"'
  },
  {
    icon: 'fa-shield',
    color: 'bullying',
    title: '霸凌防火墙',
    description: '教你识别霸凌行为，提供应对方法+求助路径，让"恶意"进不来'
  },
  {
    icon: 'fa-users',
    color: 'relationship',
    title: '关系保护伞',
    description: '解决亲子、同学间的小矛盾，给你"好好沟通"的小妙招，让关系更舒服'
  }
];

// 监听滚动事件
onMounted(() => {
  const handleScroll = () => {
    isScrolled.value = window.scrollY > 10;
  };
  
  window.addEventListener('scroll', handleScroll);
  
  // 清理函数
  return () => {
    window.removeEventListener('scroll', handleScroll);
  };
});

// 方法
const scrollToFeatures = () => {
  document.getElementById('features').scrollIntoView({
    behavior: 'smooth'
  });
  mobileMenuOpen.value = false;
};

const handleNavigation = (path) => {
  if (path === '首页') {
    window.scrollTo({
      top: 0,
      behavior: 'smooth'
    });
  } else {
    // 在实际应用中，这里会导航到对应的路由
    console.log('导航到:', path);
    // router.push(`/${path}`);
  }
  mobileMenuOpen.value = false;
};

const handleFeatureClick = (title) => {
  // 在实际应用中，这里会导航到对应的功能页面
  console.log('进入功能:', title);
  // router.push(`/${title}`);
};
</script>
<style scoped>
.home {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}</style>
