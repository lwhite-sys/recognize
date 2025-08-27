<script setup lang="ts">
import { computed } from 'vue';
import { useAppStore } from '@/store/modules/app';
const appStore = useAppStore();

const gap = computed(() => (appStore.isMobile ? 0 : 16));
</script>

<template>
  <NSpace vertical :size="16">
    <NGrid :x-gap="gap" :y-gap="16" responsive="screen" item-responsive>
        <NGi span="24 s:24 m:14">
        <NCard :bordered="false" class="card-wrapper">
           专注力防护盾【未完成，待开发】
        </NCard>
      </NGi>
    </NGrid>
  </NSpace>
</template>

<style scoped></style>
